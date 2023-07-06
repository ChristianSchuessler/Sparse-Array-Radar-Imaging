import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

"""
This file implements a matched filter to pre-process the ground-truth data.
A detailed description of the algorithm can be found in:

C. Schüßler, M. Hoffmann, I. Ullmann, R. Ebelt and M. Vossiek, 
"Deep Learning Based Image Enhancement for Automotive Radar Trained With an Advanced Virtual Sensor," 
in IEEE Access, vol. 10, pp. 40419-40431, 2022, doi: 10.1109/ACCESS.2022.3166227,

which implemented the algorithm described in:
S. S. Ahmed, A. Schiessl, F. Gumbmann, M. Tiebout, S. Methfessel and L. -P. Schmidt, 
"Advanced Microwave Imaging," in IEEE Microwave Magazine, vol. 13, no. 6, pp. 26-43, Sept.-Oct. 2012, doi: 10.1109/MMM.2012.2205772.
"""

class SignalData:
    """
    This class stores meta data of the if-signal (or beat signal)
    and also the if signal values for all antenna combinations itself
    """

    def __init__(self):
        self.tx_positions = None
        self.rx_positions = None

        self.time_vector = None
        self.carrier_frequency = None
        self.bandwidth = None
        self.chirp_duration = None
        self.cycle_duration = None
        self.delta_frequency = None
        self.signal_type = "FMCW"
        # numpy array with shape (n_tx, n_rx, n_chirp, n_time)
        self.signals = None
        
    # add some get properties for convenience
    @property
    def number_tx(self):
        return self.signals.shape[0]

    @property
    def number_rx(self):
        return self.signals.shape[1]

    @property
    def number_chirps(self):
        return self.signals.shape[2]


def holo_reconstruction_range_angle_cuda(
    range_positions, angle_positions, signal_data, z_pos=0.0, 
    chirp_index=0, zero_padding_factor=4, transformation_matrix=np.identity(4)):
    """
    Applies a holographic/matched filter reconstruction in range, sin(angle) space,
    whereby the angle_position is in sine space -> therefore it takes not angles but sin(angles)

    The z_pos indicates in which slice in cartesian coordinates the reconstruction in polar coordinates should be performed.
    A transformation_matrix can be given, which is applied to all (cartesian converted) positions before reconstruction.

    The signal is padded by a zero_padding_factor before an FFT is applied.
    Any window function has to be applied by the user in beforehand.
    """

    mod = SourceModule("""
    # define _USE_MATH_DEFINES
    #include <cuComplex.h>
    #include <math.h>
    #include <cuda.h>
    #include <stdio.h>
    __device__ __forceinline__ cuComplex comp_exp (float phase)
    {
        cuComplex result = make_cuComplex(cos(phase), sin(phase));
        return result;
    }

    __device__ float cargf(const cuComplex& z)
    {   
        return atan2(cuCimagf(z), cuCrealf(z));
    }

    __device__ float cabsf(const cuComplex& z)
    {   
        return sqrtf(cuCimagf(z)*cuCimagf(z) +  cuCrealf(z)*cuCrealf(z));
    }

    __device__ float3 operator-(const float3 &a, const float3 &b)
    {
        return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
    }

    __device__ __forceinline__ float vec_length(const float3 &a)
    {
        return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    }

    __device__ cuComplex operator*(const cuComplex &a, const cuComplex &b)
    {
        return make_cuComplex(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
    }

    __device__ cuComplex interpolate_complex(float x0, float x1, cuComplex f0, cuComplex f1, float x)
    {
        float phase0 = cargf(f0);
        float phase1 = cargf(f1);

        if (phase1 < phase0)
            phase1 += 2*M_PI;

        float mag_interp = cabsf(f0) + (cabsf(f1) - cabsf(f0))/(x1 - x0) * (x-x0);
        float phase_interp =  phase0 + ((phase1 -phase0)/(x1 - x0)) * (x-x0);
        return make_cuComplex(mag_interp*cos(phase_interp), mag_interp*sin(phase_interp));
    }

    __global__ void holo_reco
    (
        float3* tx_antennas, 
        int num_tx_antennas, 
        float3* rx_antennas,
        int num_rx_antennas,
        float* trans_matrix,
        float* range_positions,
        int num_range_positions,
        float* angle_positions,
        int num_angle_positions,
        cuComplex* reco_image,
        cuComplex* frequency_signal,
        int frequency_signal_length,
        float frequency_slope,
        float delta_frequency,
        float carrier_frequency,
        float z_pos
    )
    {
        int index_range = threadIdx.x + blockIdx.x*blockDim.x;
        int index_angle = threadIdx.y + blockIdx.y*blockDim.y;

        if (index_range >= num_range_positions || index_angle >= num_angle_positions)
            return;
        
        const float c = 3e8;
        float range_pos = range_positions[index_range];
        float angle_pos = angle_positions[index_angle];

        // convert to cartesian and transform
        float3 reco_pos_orig;
        reco_pos_orig.y = range_pos*cosf(asinf(angle_pos));
        reco_pos_orig.x = range_pos*angle_pos;

        // apply matrix
        float3 reco_pos;
        reco_pos.x = trans_matrix[0]*reco_pos_orig.x + trans_matrix[1]*reco_pos_orig.y + trans_matrix[3];
        reco_pos.y = trans_matrix[4]*reco_pos_orig.x + trans_matrix[5]*reco_pos_orig.y + trans_matrix[7];
        reco_pos.z = z_pos;

        cuComplex result = make_cuComplex(1e-3,1e-3);
        for(int tx_index = 0; tx_index < num_tx_antennas; tx_index++)
        {
            float3 tx_pos = tx_antennas[tx_index];
            for(int rx_index = 0; rx_index < num_rx_antennas; rx_index++)
            {
                float3 rx_pos = rx_antennas[rx_index];

                float delay = (vec_length(reco_pos-tx_pos) + vec_length(reco_pos-rx_pos))/c;
                float exp_phase = -2*M_PI*carrier_frequency*delay;
                cuComplex weight = comp_exp(exp_phase);

                float x = (delay*frequency_slope);
                int signal_index =  (int)(delay*frequency_slope / delta_frequency);

                int x0_array_idx = tx_index*num_rx_antennas*frequency_signal_length + rx_index*frequency_signal_length + signal_index;
                int x1_array_idx = tx_index*num_rx_antennas*frequency_signal_length + rx_index*frequency_signal_length + (signal_index+1);
                
                cuComplex f0 = frequency_signal[x0_array_idx];
                cuComplex f1 = frequency_signal[x1_array_idx];

                float x0 = signal_index*delta_frequency;
                float x1 = (signal_index+1)*delta_frequency;
                cuComplex signal_value = interpolate_complex(x0, x1, f0, f1, x);

                cuComplex part_result =signal_value*weight;
                result.x = result.x + part_result.x;
                result.y = result.y + part_result.y;
            }
        }
        int result_index = index_angle*num_range_positions + index_range;
        reco_image[result_index] = result;
    }
    """)

    bandwidth = signal_data.bandwidth
    chirp_duration = signal_data.chirp_duration

    frequency_slope = np.float32(bandwidth/chirp_duration)
    delta_frequency = np.float32(1.0/(chirp_duration*zero_padding_factor))

    tx_antennas = np.asarray(signal_data.tx_positions)
    rx_antennas = np.asarray(signal_data.rx_positions)
    
    reco_trans_matrix = transformation_matrix.astype(np.float32)
    reco_positions_range = range_positions.astype(np.float32)
    reco_positions_angle = angle_positions.astype(np.float32)

    num_tx_antennas = np.int32(len(tx_antennas))
    num_rx_antennas = np.int32(len(rx_antennas))

    tx_antennas = np.ravel(tx_antennas.astype(np.float32))
    rx_antennas = np.ravel(rx_antennas.astype(np.float32))

    reco_image = 1e-9*np.ones((reco_positions_range.shape[0], reco_positions_angle.shape[0]), dtype=np.complex64)
    reco_image = np.ravel(reco_image)
    raw_signal = signal_data.signals

    # create frequency signal
    frequency_signal = np.empty((raw_signal.shape[0], raw_signal.shape[1], raw_signal.shape[3]*zero_padding_factor), dtype=np.complex64)
    for tx_index in range(len(signal_data.tx_positions)):
        for rx_index in range(len(signal_data.rx_positions)):
            zero_padded_signal = np.zeros(frequency_signal.shape[2], dtype=np.complex128)
            zero_padded_signal[:raw_signal.shape[3]] = raw_signal[tx_index, rx_index, chirp_index]
            frequency_signal[tx_index, rx_index] = np.fft.fft(zero_padded_signal)

    frequency_signal_length = np.int32(frequency_signal.shape[2])

    # copy data to gpu
    reco_trans_matrix_gpu = cuda.mem_alloc(reco_trans_matrix.nbytes)
    reco_positions_range_gpu = cuda.mem_alloc(reco_positions_range.nbytes)
    reco_positions_angle_gpu = cuda.mem_alloc(reco_positions_angle.nbytes)

    tx_antennas_gpu = cuda.mem_alloc(tx_antennas.nbytes)
    rx_antennas_gpu = cuda.mem_alloc(rx_antennas.nbytes)
    reco_image_gpu = cuda.mem_alloc(reco_image.nbytes)
    frequency_signal = np.ravel(frequency_signal)
    frequency_signal_gpu = cuda.mem_alloc(frequency_signal.nbytes)
    
    cuda.memcpy_htod(reco_trans_matrix_gpu, reco_trans_matrix)
    cuda.memcpy_htod(tx_antennas_gpu, tx_antennas)
    cuda.memcpy_htod(rx_antennas_gpu, rx_antennas)
    cuda.memcpy_htod(reco_image_gpu, reco_image)
    cuda.memcpy_htod(frequency_signal_gpu, frequency_signal)
    cuda.memcpy_htod(reco_positions_range_gpu, reco_positions_range)
    cuda.memcpy_htod(reco_positions_angle_gpu, reco_positions_angle)

    num_reco_positions_range = np.int32(len(reco_positions_range))
    num_reco_positions_angle = np.int32(len(reco_positions_angle))
    carrier_frequency = np.float32(signal_data.carrier_frequency)

    func = mod.get_function("holo_reco")
    func(
    tx_antennas_gpu,\
    num_tx_antennas, \
    rx_antennas_gpu,\
    num_rx_antennas,\
    reco_trans_matrix_gpu,\
    reco_positions_range_gpu,\
    num_reco_positions_range,\
    reco_positions_angle_gpu,\
    num_reco_positions_angle,\
    reco_image_gpu,\
    frequency_signal_gpu,
    frequency_signal_length,\
    frequency_slope,\
    delta_frequency,\
    carrier_frequency,\
    np.float32(z_pos),\
    block=(8, 8, 1), grid=(128, 128, 1))
    cuda.memcpy_dtoh(reco_image, reco_image_gpu)
    reco_image = reco_image.reshape((reco_positions_angle.shape[0], reco_positions_range.shape[0]))
    reco_image = reco_image / (num_rx_antennas*num_tx_antennas)

    return reco_image

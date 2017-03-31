#if CONFIG_USE_DOUBLE

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define DOUBLE_SUPPORT_AVAILABLE
#endif

#endif // CONFIG_USE_DOUBLE

#if defined(DOUBLE_SUPPORT_AVAILABLE)

// double
typedef double real_t;
typedef double2 real2_t;
typedef double3 real3_t;
typedef double4 real4_t;
typedef double8 real8_t;
typedef double16 real16_t;
#define PI 3.14159265358979323846

#else

// float
typedef float real_t;
typedef float2 real2_t;
typedef float3 real3_t;
typedef float4 real4_t;
typedef float8 real8_t;
typedef float16 real16_t;
#define PI 3.14159265359f

#endif

__kernel void vadd(__global const real_t* a, __global const real_t* b, __global real_t* c, int n) {
    // int id = get_global_id(0);
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = a[id] + b[id];
}

__kernel void vaddscalar(__global const float* a, const float b, __global float* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = a[id] + b;
}

__kernel void vsubtract(__global const real_t* a, __global const real_t* b, __global real_t* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = a[id] - b[id];
}

__kernel void vsubtractscalar(__global const float* a, const float b, __global float* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = a[id] - b;
}

__kernel void scalarsubtractv(__global const float* a, const float b, __global float* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = b - a[id];
}

#define TILE_WIDTH 10

// Compute C = A * B
__kernel void vdot(__global const float *A, __global const float *B, __global float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TILE_WIDTH*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TILE_WIDTH*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TILE_WIDTH][TILE_WIDTH];
    __local float Bsub[TILE_WIDTH][TILE_WIDTH];
 
    // Initialise the accumulation register
    float acc = 0.0f;
    
    // Loop over all tiles
    const int numTiles = numAColumns / TILE_WIDTH;
    for (int t = 0; t<numTiles; t++) {
	    // Load one tile of A and B into local memory
        const int tiledRow = TILE_WIDTH*t + row;
        const int tiledCol = TILE_WIDTH*t + col;
        Asub[col][row] = A[tiledCol*numARows + globalRow];
        Bsub[col][row] = B[globalCol*numAColumns + tiledRow];
 
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            acc += Asub[k][row] * Bsub[col][k];
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final result in C
    C[globalCol*numARows + globalRow] = acc;
}

/* __kernel void vdot(__global const float *A, __global const float *B, __global float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
    // 2D Thread ID
    // Old CUDA code
    //int tx = blockIdx.x * TILE_SIZE + threadIdx.x;
    //int ty = blockIdx.y * TILE_SIZE + threadIdx.y;
    int tx = get_global_id(0); 
    int ty = get_global_id(1);
 
    // value stores the element that is 
    // computed by the thread
    float value = 0;
    for (int k = 0; k < numAColumns; ++k)
    {
        float elementA = A[ty * numAColumns + k];
        float elementB = B[k * numCColumns + tx];
        value += elementA * elementB;
    }
 
    // Write the matrix to device memory each 
    // thread writes one element
    C[ty * numAColumns + tx] = value;
} */

__kernel void vmul(__global const float* a, __global const float* b, __global float* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = a[id] * b[id];
}

__kernel void vmulscalar(__global const float* a, const float b, __global float* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = a[id] * b;
}

__kernel void vdivide(__global const float* a, __global const float* b, __global float* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = a[id] / b[id];
}

__kernel void vdividescalar(__global const float* a, const float b, __global float* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = a[id] / b;
}

__kernel void scalardividev(__global const float* a, const float b, __global float* c, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	c[id] = b / a[id];
}

/* __kernel void vsum(__global float *a, __global float *r, __local float *b) {
    uint gid = get_global_id(0);
    uint wid = get_group_id(0);
    uint lid = get_local_id(0);
    uint gs = get_local_size(0);

    b[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint s = gs/2; s > 0; s >>= 1) {
        if(lid < s) {
          b[lid] += b[lid+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(lid == 0) r[wid] = b[lid];
} */

__kernel void vsum(__global const float *input, __global float *output, __local float* reductionSums) {
	const int globalID = get_global_id(0);
	const int localID = get_local_id(0);
	const int localSize = get_local_size(0);
	const int workgroupID = globalID / localSize;

	int inputSize = get_global_size(0);
	// __local float reductionSums[32];

	reductionSums[localID] = input[globalID];
	
	for(int offset = localSize / 2; offset > 0; offset /= 2) {
		barrier(CLK_LOCAL_MEM_FENCE);	// wait for all other work-items to finish previous iteration.
		if(localID < offset) {
			reductionSums[localID] += reductionSums[localID + offset];
		}
	}
	
	if(localID == 0) {	// the root of the reduction subtree
		output[workgroupID] = reductionSums[0];
	}
}


__kernel void vset(__global float* a, __global const float* b, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	a[id] = b[id];
}

__kernel void vsetscalar(__global float* a, const float b, int n) {
	int id = get_global_id(1) * get_global_size(0) + get_global_id(0);
	a[id] = b;
}


// forward1.cl
__kernel void convolve_forward(const int numExamples, __global const float *inputs, __global const float * filters, __global float *output,
    int gInputSize, int gInputSizeSquared, int gNumFilters, int gFilterSize, int gFilterSizeSquared, int gHalfFilterSize, int gOutputSize, int gOutputSizeSquared, 
    int gNumInputPlanes, int gInputPlanes, int gEven
) {
    int globalId = get_global_id(0);

	int outputImage2Id = globalId / gOutputSizeSquared;
	int exampleId = outputImage2Id / gNumFilters;
	int filterId = outputImage2Id % gNumFilters;

	// Intra image coords.
	int localid = globalId % gOutputSizeSquared;
	int outputRow = localid / gOutputSize;
	int outputCol = localid % gOutputSize;
	
	global float const *inputCube = inputs + exampleId * gNumInputPlanes * gInputSizeSquared;
	global float const *filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;
	
	float sum = 0;
	if (exampleId < numExamples) {
	    for (int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++) {
		    global float const *inputPlane = inputCube + inputPlaneIdx * gInputSizeSquared;
			global float const *filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;
			for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {
			    // Trying to reduce register pressure.
				#if gPadZeros == 1
				    #define inputRowIdx (outputRow + u)
				#else
				    #define inputRowIdx (outputRow + u + gHalfFilterSize)
				#endif
				global float const *inputRow = inputPlane + inputRowIdx * gInputSize;
				global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
				bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;
				#pragma unroll
				for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {
				    #if gPadZeros == 1
					    # define inputColIdx (outputCol + v)
					#else
					    # define inputColIdx (outputCol + v + gHalfFilterSize)
					#endif
					bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;
					if (process) {
					    sum += inputRow[inputColIdx] * filterRow[v];
					}
				}
			}
		}
	}
	
	if (exampleId < numExamples) {
	    output[globalId] = sum;
	} 
}

// bw_rowperwg.cl
__kernel void convolve_backward(
    const float learningRateMultiplier, const int batchSize,
	__global const float *gradOutput, global const float *images,
	__global float *weightChanges,
	#ifdef BIASED
	    global float *biasWeightChanges,
	#endif
	__local float *_errorImage, __local float *_imageImage, 
	int gInputSize, int gInputSizeSquared, int gNumFilters, int gFilterSize, int gFilterSizeSquared, int gOutputSize, int gOutputSizeSquared,
	int gNumInputPlanes, int gInputPlanes, int gEven, int gMargin
) {
    const int globalId = get_global_id(0);
	const int localId = get_local_id(0);
	const int workgroupId = get_group_id(0);
	const int workgroupSize = get_local_size(0);

	const int filterRow = localId / gFilterSize;
	const int filterCol = localId % gFilterSize;

	const int inputRow = workgroupId % gInputSize;
	const int outputPlane = (workgroupId / gInputSize) / gInputPlanes;
	const int inputPlane = (workgroupId / gInputSize) % gInputPlanes;

	float thiswchange = 0;
	#ifdef BIASED
	    float thisbiaschange = 0;
	#endif

	for (int n = 0; n < batchSize; n++) {
	    int upstreamImageGlobalOffset = (n * gInputPlanes + inputPlane) * gInputSizeSquared;
		// Need to fetch the image, but it's bigger than us, so will need to loop.
		const int numLoopsForUpstream = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int i = 0; i < numLoopsForUpstream; i++) {
		    int thisOffset = i * workgroupSize + localId;
			if (thisOffset < gInputSizeSquared) {
			    _imageImage[thisOffset] = images[upstreamImageGlobalOffset + thisOffset];
			}
		}
		int resultImageGlobalOffset = (n * gNumFilters + outputPlane) * gOutputSizeSquared;
		int numLoopsForOutput = (gOutputSizeSquared + workgroupSize - 1) / workgroupSize;
		for (int i = 0; i < numLoopsForOutput; i++) {
		    int thisOffset = i * workgroupSize + localId;
			if (thisOffset < gOutputSizeSquared) {
			    _errorImage[thisOffset] = gradOutput[resultImageGlobalOffset + thisOffset];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (localId < gFilterSizeSquared) {
		    for (int outRow = 0; outRow < gOutputSize; outRow++) {
			    int inputRow = outRow - gMargin + filterRow;
				for (int outCol = 0; outCol < gOutputSize; outCol++) {
				    int inputCol = outCol - gMargin + filterCol;
					bool proceed = inputRow >= 0 && inputCol >= 0 && inputRow < gInputSize && inputCol < gInputSize;
					if (proceed) {
					    int resultIndex = outRow * gOutputSize + outCol;
						float error = _errorImage[resultIndex];
						int upstreamDataIndex = inputRow * gInputSize + inputCol;
						float upstreamResult = _imageImage[upstreamDataIndex];
						thiswchange += upstreamResult * error;

						#ifdef BIASED
						    thisbiaschange += error;
						#endif
					}
				}
			}
		}
	}

	/* if (localId < gFilterSizeSquared) {
	    weights[workgroupId * gFilterSizeSquared + localId] -= learningRateMultiplier * thiswchange;
	}

	#ifdef BIASED
	    bool writeBias = inputPlane == 0 && localId == 0;
		if (writeBias) {
		    biasWeights[outputPlane] -= learningRateMultiplier * thisbiaschange;
		}
	#endif */

}
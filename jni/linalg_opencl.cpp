#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include "linalg_opencl.h"

#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <CL/cl.hpp>
#include <algorithm>


static cl::Context      gContext;
static cl::CommandQueue gQueue;
static cl::Kernel       vaddKernel;
static cl::Kernel       vaddScalarKernel;
static cl::Kernel       vsubtractKernel;
static cl::Kernel       vsubtractScalarKernel;
static cl::Kernel       scalarSubtractVKernel;
static cl::Kernel       vdotKernel;
static cl::Kernel       vmulKernel;
static cl::Kernel       vmulScalarKernel;
static cl::Kernel       vdivideKernel;
static cl::Kernel       vdivideScalarKernel;
static cl::Kernel       scalarDivideVKernel;
static cl::Kernel       vsumKernel;

char *file_contents(const char *filename, int *length)
{
    FILE *f = fopen(filename, "r");
    void *buffer;

    if (!f) {
        LOGE("Unable to open %s for reading\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);

    buffer = malloc(*length+1);
    *length = fread(buffer, 1, *length, f);
    fclose(f);
    ((char*)buffer)[*length] = '\0';

    return (char*)buffer;
}

bool throwJavaException(JNIEnv *env,std::string method_name,std::string exception_msg, int errorCode=0)
{
    char buf[8];
    sprintf(buf,"%d",errorCode);
    std::string code(buf);

    std::string msg = "@" + method_name + ": " + exception_msg + " ";
    if(errorCode!=0) msg += code;

    jclass generalExp = env->FindClass("java/lang/Exception");
    if (generalExp != 0) {
        env->ThrowNew(generalExp, msg.c_str());
        return true;
    }
    return false;
}

void cb(cl_program p,void* data)
{
    clRetainProgram(p);
    cl_device_id devid[1];
    clGetProgramInfo(p,CL_PROGRAM_DEVICES,sizeof(cl_device_id),(void*)devid,NULL);
    char bug[65536];
    clGetProgramBuildInfo(p,devid[0],CL_PROGRAM_BUILD_LOG,65536*sizeof(char),bug,NULL);
    clReleaseProgram(p);
    LOGE("Build log \n %s\n",bug);
}

JNIEXPORT jboolean JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_compileKernels(JNIEnv *env, jclass clazz)
{
    // Find OCL devices and compile kernels
    cl_int err = CL_SUCCESS;
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            return false;
        }
        cl_context_properties properties[] =
        { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
        gContext = cl::Context(CL_DEVICE_TYPE_GPU, properties);
        std::vector<cl::Device> devices = gContext.getInfo<CL_CONTEXT_DEVICES>();
        gQueue = cl::CommandQueue(gContext, devices[0], 0, &err);
        int src_length = 0;
		const char* src  = file_contents("/data/data/en.menghui.android.linearalgebraparallel/app_execdir/kernels.cl",&src_length);
        cl::Program::Sources sources(1,std::make_pair(src, src_length) );
        cl::Program program(gContext, sources);
        program.build(devices,NULL,cb);
        while(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) != CL_BUILD_SUCCESS);
        vaddKernel = cl::Kernel(program, "vadd", &err);
        vaddScalarKernel = cl::Kernel(program, "vaddscalar", &err);
        vsubtractKernel = cl::Kernel(program, "vsubtract", &err);
        vsubtractScalarKernel = cl::Kernel(program, "vsubtractscalar", &err);
        scalarSubtractVKernel = cl::Kernel(program, "scalarsubtractv", &err);
        vdotKernel = cl::Kernel(program, "vdot", &err);
        vmulKernel = cl::Kernel(program, "vmul", &err);
		vmulScalarKernel = cl::Kernel(program, "vmulscalar", &err);
		vdivideKernel = cl::Kernel(program, "vdivide", &err);
		vdivideScalarKernel = cl::Kernel(program, "vdividescalar", &err);
		scalarDivideVKernel = cl::Kernel(program, "scalardividev", &err);
		vsumKernel = cl::Kernel(program, "vsum", &err);

        return true;
    }
    catch (cl::Error e) {
        if( !throwJavaException(env,"decode",e.what(),e.err()) )
            LOGI("@decode: %s \n",e.what());
        return false;
    }
}

/* void helper(double* out, int osize, double* inA, double* inB int isize, int w, int h, int choice)
{
    try {
        cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, isize*sizeof(double), inA, NULL);
        cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, osize*sizeof(double));
        cl::Buffer bufferOut2= cl::Buffer(gContext, CL_MEM_READ_WRITE, osize*sizeof(double));
        vmulKernel.setArg(2,w);
        vmulKernel.setArg(3,h);
        vmulKernel.setArg(1,bufferIn);
        vmulKernel.setArg(0,bufferOut);
        gQueue.enqueueNDRangeKernel(vmulKernel,
                cl::NullRange,
                cl::NDRange( (int)ceil((float)w/16.0f)*16,(int)ceil((float)h/16.0f)*16),
                cl::NDRange(16,16),
                NULL,
                NULL);
        if (choice>0) {
        	vaddKernel.setArg(2,w);
        	vaddKernel.setArg(3,h);
        	vaddKernel.setArg(1,bufferOut);
        	vaddKernel.setArg(0,bufferOut2);
            gQueue.enqueueNDRangeKernel(vaddKernel,
                    cl::NullRange,
                    cl::NDRange( (int)ceil((float)w/16.0f)*16,(int)ceil((float)h/16.0f)*16),
                    cl::NDRange(16,16),
                    NULL,
                    NULL);
        }
        gQueue.enqueueReadBuffer(bufferOut2, CL_TRUE, 0, osize*sizeof(cl_uchar4), out);
    }
    catch (cl::Error e) {
        LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
    }
} */

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_addVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloatArray inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    // double *inDataA = (*env)->GetDoubleArrayElements(env, inDataAJava, NULL);
    // double *inDataB = (*env)->GetDoubleArrayElements(env, inDataBJava, NULL);
    // double *outData = (*env)->GetDoubleArrayElements(env, outDataJava, NULL);

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    /* float inDataAFloat[insize];
    float inDataBFloat[insize];
    float outDataFloat[outsize];

    std::copy(inDataA, inDataA + insize, inDataAFloat);
    std::copy(inDataB, inDataB + insize, inDataBFloat);
    std::copy(outData, outData + outsize, outDataFloat); */

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		vaddKernel.setArg(0,bufferA);
		vaddKernel.setArg(1,bufferB);
		vaddKernel.setArg(2,bufferOut);
		vaddKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vaddKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

	// std::copy(outDataFloat, outDataFloat + outsize, outData);

    // release the memory so java can have it again
    // (*env)->ReleaseDoubleArrayElements(env, outDataJava, outData, 0);
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_addScalarVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloat inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    // float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		// cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		vaddScalarKernel.setArg(0,bufferA);
		// vaddScalarKernel.setArg(1,bufferB);
		vaddScalarKernel.setArg(1,inDataBJava);
		vaddScalarKernel.setArg(2,bufferOut);
		vaddScalarKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vaddScalarKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    // env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_subtractVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloatArray inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		vsubtractKernel.setArg(0,bufferA);
		vsubtractKernel.setArg(1,bufferB);
		vsubtractKernel.setArg(2,bufferOut);
		vsubtractKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vsubtractKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_subtractScalarVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloat inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    // float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		// cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		vsubtractScalarKernel.setArg(0,bufferA);
		// vsubtractScalarKernel.setArg(1,bufferB);
		vsubtractScalarKernel.setArg(1,inDataBJava);
		vsubtractScalarKernel.setArg(2,bufferOut);
		vsubtractScalarKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vsubtractScalarKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    // env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_scalarSubtractVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloat inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    // float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		// cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		scalarSubtractVKernel.setArg(0,bufferA);
		// scalarSubtractVKernel.setArg(1,bufferB);
		scalarSubtractVKernel.setArg(1,inDataBJava);
		scalarSubtractVKernel.setArg(2,bufferOut);
		scalarSubtractVKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(scalarSubtractVKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    // env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_dotVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloatArray inDataBJava,
		jfloatArray outDataJava,
        jint aWidth,
        jint aHeight,
		jint bWidth,
		jint bHeight,
		jint outWidth,
		jint outHeight)
{
    int aSize = aWidth * aHeight;
    int bSize = bWidth * bHeight;
    int outSize = outWidth * outHeight;

    int const TILE_WIDTH = 10;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, aSize*sizeof(float), inDataA, NULL);
		cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bSize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outSize*sizeof(float));

		vdotKernel.setArg(0,bufferA);
		vdotKernel.setArg(1,bufferB);
		vdotKernel.setArg(2,bufferOut);
		vdotKernel.setArg(3,aHeight);
		vdotKernel.setArg(4,aWidth);
		vdotKernel.setArg(5,bHeight);
		vdotKernel.setArg(6,bWidth);
		vdotKernel.setArg(7,outHeight);
		vdotKernel.setArg(8,outWidth);

		if (outWidth > TILE_WIDTH && outHeight > TILE_WIDTH) {
			// gQueue.enqueueNDRangeKernel(vdotKernel, cl::NullRange, cl::NDRange((outWidth-1)/TILE_WIDTH+1, (outHeight-1)/TILE_WIDTH+1), cl::NDRange(TILE_WIDTH, TILE_WIDTH), NULL, NULL);

			gQueue.enqueueNDRangeKernel(vdotKernel,
								cl::NullRange,
								cl::NDRange(aHeight, bWidth),
								cl::NDRange(TILE_WIDTH, TILE_WIDTH),
								NULL,
								NULL);
		} else {
			gQueue.enqueueNDRangeKernel(vdotKernel,
					cl::NullRange,
					cl::NDRange(aHeight, bWidth),
					cl::NullRange,
					NULL,
					NULL);
		}

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outSize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}


JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_dot(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloatArray inDataBJava,
		jfloatArray outDataJava,
        jint aWidth,
        jint aHeight,
		jint bWidth,
		jint bHeight,
		jint outWidth,
		jint outHeight)
{
    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    /* for (int i = 0; i < aHeight; i++) {
    	for (int j = 0; j < bWidth; j++) {
    		for (int k = 0; k < aWidth; k++) {
    			outData[i * bWidth + j] += inDataA[i * aWidth + k] * inDataB[k * bWidth + j];
    		}
    	}
    } */

    float Bcolj[bWidth];
	for (int j = 0; j < bWidth; j++) {
		for (int k = 0; k < aWidth; k++)
			Bcolj[k] = inDataB[k * bWidth + j];

		for (int i = 0; i < aHeight; i++) {
			float s = 0;
			for (int k = 0; k < aWidth; k++)
				s += inDataA[i * aWidth + k] * Bcolj[k];
			outData[j * bWidth + i] = s;
		}
	}

    /* int block_size = 50;
    for (int i = 0; i < aHeight; i += block_size) {
		for (int j = 0; j < bWidth; j += block_size) {
			for (int k = 0; k < aWidth; k += block_size) {
				// B * B mini matrix multiplications.
				for (int i1 = i; i1 < i + block_size; i1++) {
					for (int j1 = j; j1 < j + block_size; j1++) {
						for (int k1 = k; k1 < k + block_size; k1++) {
							outData[i1 * bWidth + j1] += inDataA[i1 * aWidth + k1] * inDataB[k1 * bWidth + j1];
						}
					}
				}
			}
		}
	} */

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

int min(int a, int b) {
	if (a < b) {
		return a;
	} else {
		return b;
	}
}


JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_mulVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloatArray inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		vmulKernel.setArg(0,bufferA);
		vmulKernel.setArg(1,bufferB);
		vmulKernel.setArg(2,bufferOut);
		vmulKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vmulKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_mulScalarVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloat inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    // float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		// cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		vmulScalarKernel.setArg(0,bufferA);
		// vmulScalarKernel.setArg(1,bufferB);
		vmulScalarKernel.setArg(1,inDataBJava);
		vmulScalarKernel.setArg(2,bufferOut);
		vmulScalarKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vmulScalarKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    // env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_divideVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloatArray inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		vdivideKernel.setArg(0,bufferA);
		vdivideKernel.setArg(1,bufferB);
		vdivideKernel.setArg(2,bufferOut);
		vdivideKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vdivideKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_divideScalarVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloat inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    // float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    size_t Bsz = 64;
    size_t Gsz = insize-1/(float)Bsz+1;

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		// cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		vdivideScalarKernel.setArg(0,bufferA);
		// vsubtractScalarKernel.setArg(1,bufferB);
		vdivideScalarKernel.setArg(1,inDataBJava);
		vdivideScalarKernel.setArg(2,bufferOut);
		vdivideScalarKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vdivideScalarKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    // env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloatArray JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_scalarDivideVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
		jfloat inDataBJava,
		jfloatArray outDataJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    // float* inDataB = (float*)env->GetPrimitiveArrayCritical(inDataBJava, 0);
    float* outData = (float*)env->GetPrimitiveArrayCritical(outDataJava, 0);

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		// cl::Buffer bufferB = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataB, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, outsize*sizeof(float));

		scalarDivideVKernel.setArg(0,bufferA);
		// scalarDivideVKernel.setArg(1,bufferB);
		scalarDivideVKernel.setArg(1,inDataBJava);
		scalarDivideVKernel.setArg(2,bufferOut);
		scalarDivideVKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(scalarDivideVKernel,
				cl::NullRange,
				cl::NDRange(width, height),
				cl::NullRange,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, outsize*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);
    // env->ReleasePrimitiveArrayCritical(inDataBJava, inDataB, 0);
    env->ReleasePrimitiveArrayCritical(outDataJava, outData, 0);

    return outDataJava;
}

JNIEXPORT jfloat JNICALL Java_en_menghui_android_linearalgebraparallel_maths_MatrixParallel_sumVector(
        JNIEnv *env,
		jclass clazz,
        jfloatArray inDataAJava,
        jint width,
        jint height)
{
    int insize = width*height;
    int outsize = insize;

    int nThreads = 8;
    int multiplier = ((insize % nThreads) == 0) ? insize / nThreads : (insize / nThreads) + 1;

    float* inDataA = (float*)env->GetPrimitiveArrayCritical(inDataAJava, 0);
    float outData[multiplier];

    try {
		cl::Buffer bufferA = cl::Buffer(gContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, insize*sizeof(float), inDataA, NULL);
		cl::Buffer bufferOut = cl::Buffer(gContext, CL_MEM_READ_WRITE, multiplier*sizeof(float));

		vsumKernel.setArg(0, bufferA);
		vsumKernel.setArg(1, bufferOut);
		vsumKernel.setArg(2, nThreads*sizeof(float), NULL);
		// vsumKernel.setArg(3,insize);
		gQueue.enqueueNDRangeKernel(vsumKernel,
				cl::NullRange,
				insize,
				nThreads,
				NULL,
				NULL);

		gQueue.enqueueReadBuffer(bufferOut, CL_TRUE, 0, multiplier*sizeof(float), outData);
	} catch (cl::Error e) {
		LOGI("@oclDecoder: %s %d \n",e.what(),e.err());
	}

	float sum = 0.0f;
	for (int i = 0; i < multiplier; i++) {
		sum += outData[i];
		LOGI("\nI: %f", outData[i]);
	}
	/* for (int i = 0; i < insize; i++) {
		sum += inDataA[i];
	} */

    // release the memory so java can have it again
    env->ReleasePrimitiveArrayCritical(inDataAJava, inDataA, 0);

    return sum;
}

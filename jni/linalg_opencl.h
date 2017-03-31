#ifndef __JNI_H__
#define __JNI_H__
#ifdef __cplusplus

#include <android/bitmap.h>
#include <android/log.h>

#define app_name "JNIProcessor"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, app_name, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, app_name, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, app_name, __VA_ARGS__))
extern "C" {
#endif

#define DECLARE_NOPARAMS(returnType,fullClassName,func) \
JNIEXPORT returnType JNICALL Java_##fullClassName##_##func(JNIEnv *env, jclass clazz);

#define DECLARE_WITHPARAMS(returnType,fullClassName,func,...) \
JNIEXPORT returnType JNICALL Java_##fullClassName##_##func(JNIEnv *env, jclass clazz,__VA_ARGS__);

DECLARE_NOPARAMS(jboolean,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,compileKernels)

DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,addVector,jfloatArray inDataAJava, jfloatArray inDataBJava, jfloatArray outDataJava, jint width, jint height)
DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,addScalarVector,jfloatArray inDataAJava, jfloat inDataBJava, jfloatArray outDataJava, jint width, jint height)

DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,subtractVector,jfloatArray inDataAJava, jfloatArray inDataBJava, jfloatArray outDataJava, jint width, jint height)
DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,subtractScalarVector,jfloatArray inDataAJava, jfloat inDataBJava, jfloatArray outDataJava, jint width, jint height)
DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,scalarSubtractVector,jfloatArray inDataAJava, jfloat inDataBJava, jfloatArray outDataJava, jint width, jint height)

DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,dotVector,jfloatArray inDataAJava, jfloatArray inDataBJava, jfloatArray outDataJava, jint aWidth, jint aHeight, jint bWidth, jint bHeight, jint outWidth, jint outHeight)

DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,mulVector,jfloatArray inDataAJava, jfloatArray inDataBJava, jfloatArray outDataJava, jint width, jint height)
DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,mulScalarVector,jfloatArray inDataAJava, jfloat inDataBJava, jfloatArray outDataJava, jint width, jint height)

DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,divideVector,jfloatArray inDataAJava, jfloatArray inDataBJava, jfloatArray outDataJava, jint width, jint height)
DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,divideScalarVector,jfloatArray inDataAJava, jfloat inDataBJava, jfloatArray outDataJava, jint width, jint height)
DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,scalarDivideVector,jfloatArray inDataAJava, jfloat inDataBJava, jfloatArray outDataJava, jint width, jint height)

DECLARE_WITHPARAMS(jfloat,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,sumVector,jfloatArray inDataAJava, jint width, jint height)


DECLARE_WITHPARAMS(jfloatArray,en_menghui_android_linearalgebraparallel_maths_MatrixParallel,dot,jfloatArray inDataAJava, jfloatArray inDataBJava, jfloatArray outDataJava, jint aWidth, jint aHeight, jint bWidth, jint bHeight, jint outWidth, jint outHeight)

#ifdef __cplusplus
}
#endif

#endif //__JNI_H__

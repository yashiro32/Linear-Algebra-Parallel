### L.A.P

L.A.P in short for Linear Algebra Parallel is a Linear Algebra Library using parallel compution using GPU for the Android Platform.

### Third party libraries

Uses JAMA a basic linear algebra package for Java for Matrix Multiplication benchmark comparision.
(http://math.nist.gov/javanumerics/jama/)

Uses Proprietary OpenCL drivers for Qualcomm's Adreno GPUs
(https://github.com/madeye/opencl-android-proprietary/tree/master/adreno-3xx)

Note: tested only on Android devices with Qualcomm Adreno GPUs.

### Example Code

##### Matrix Multiplication comparison. 

```
MatrixParallel mat = new MatrixParallel(Arrays.asList(1000, 1000), 2, this);
MatrixParallel b = new MatrixParallel(Arrays.asList(1000, 1000), 3, this);

Matrix matOne = new Matrix(1000, 1000, 2);
Matrix matTwo = new Matrix(1000, 1000, 3);
long start = System.currentTimeMillis();
Matrix ma = matOne.times(matTwo);

Log.d(TAG, "Time taken for java dot multiplication using JAMA: " + (System.currentTimeMillis() - start));

start = System.currentTimeMillis();
MatrixParallel result = mat.dot(b);
Log.d(TAG, "Time taken for gpu parallel dot multiplication: " + (System.currentTimeMillis() - start));
```
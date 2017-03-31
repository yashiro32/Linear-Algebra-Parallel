package en.menghui.android.linearalgebraparallel.utils;

public class ArraysUtils {
	public static float[] doubleArrayToFloatArray (double[] dArr) {
		float[] fArr = new float[dArr.length];
		for(int i = 0; i < dArr.length; i++) {
			fArr[i] = (float) dArr[i];
		}
		
		return fArr;
	}
}

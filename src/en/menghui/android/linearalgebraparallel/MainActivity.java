package en.menghui.android.linearalgebraparallel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import en.menghui.android.linearalgebraparallel.maths.MatrixParallel;
import Jama.Matrix;
import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;

public class MainActivity extends Activity {

	private static final String TAG = "Main Activity";
	
	private Matrix matOne;
	private Matrix matTwo;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		TextView tv1 = (TextView) findViewById(R.id.tv1);
		
		MatrixParallel mat = new MatrixParallel(Arrays.asList(1000, 1000), 2, this);
		/* mat.tmat.set(0, 0, 10.0);
		mat.tmat.set(0, 1, 20.0);
		mat.tmat.set(0, 2, 30.0);
		mat.tmat.set(1, 0, 40.0);
		mat.tmat.set(1, 1, 50.0);
		mat.tmat.set(1, 2, 60.0);
		mat.tmat.set(2, 0, 70.0);
		mat.tmat.set(2, 1, 80.0);
		mat.tmat.set(2, 2, 90.0); */
		MatrixParallel b = new MatrixParallel(Arrays.asList(1000, 1000), 3, this);
		/* b.tmat.set(0, 0, 1.0);
		b.tmat.set(0, 1, 2.0);
		b.tmat.set(0, 2, 3.0);
		b.tmat.set(1, 0, 4.0);
		b.tmat.set(1, 1, 5.0);
		b.tmat.set(1, 2, 6.0);
		b.tmat.set(2, 0, 7.0);
		b.tmat.set(2, 1, 8.0);
		b.tmat.set(2, 2, 9.0); */
		
		matOne = new Matrix(1000, 1000, 2);
		matTwo = new Matrix(1000, 1000, 3);
		long start = System.currentTimeMillis();
		Matrix ma = matOne.plus(matTwo);
		
		// float[] res = new float[mat.shape.get(0) * b.shape.get(1)];
		/* for (int i = 0; i < mat.shape.get(0); i++) {
			for (int j = 0; j < b.shape.get(1); j++) {
				for (int k = 0; k < mat.shape.get(1); k++) {
					res[i * b.shape.get(1) + j] += mat.tarr[i * mat.shape.get(1) + k] * b.tarr[k * b.shape.get(1) + j];
		    	}
		    }
		} */
		
		/* int block_size = 50;
	    for (int i = 0; i < mat.shape.get(0); i += block_size) {
			for (int j = 0; j < b.shape.get(1); j += block_size) {
				for (int k = 0; k < mat.shape.get(1); k += block_size) {
					// B * B mini matrix multiplications.
					for (int i1 = i; i1 < i + block_size; i1++) {
						for (int j1 = j; j1 < j + block_size; j1++) {
							for (int k1 = k; k1 < k + block_size; k1++) {
								res[i1 * b.shape.get(1) + j1] += mat.tarr[i1 * mat.shape.get(1) + k1] * b.tarr[k1 * b.shape.get(1) + j1];
							}
						}
					}
				}
			}
		} */
		
		Log.d(TAG, "Time taken for java dot multiplication: " + (System.currentTimeMillis() - start));
		
		start = System.currentTimeMillis();
		MatrixParallel result = mat.add(b);
		Log.d(TAG, "Time taken for gpu parallel dot multiplication: " + (System.currentTimeMillis() - start));
		// Log.d(TAG, "Result shape 0: " + result.shape.get(0) + " Result shape 1: " + result.shape.get(1) + " value: " + result.tarr[result.tarr.length-1]);
		// Matrix result = mat.divide(3.0, mat.tmat);
		
		// Matrix result = mat.dot(mat.thisMatrix, b);
		// Matrix result = mat.thisMatrix.times(b);
		
		/* for (int i = 0; i < result.tarr.length; i++) {
			// Log.d(TAG, "i: " + i + " j: " + j + " val: " + result.get(i, j));
			tv1.append("\ni: " + i + " val: " + result.tarr[i]);
		} */
		
		/* start = System.currentTimeMillis();
		float sum = result.sum();
		Log.d(TAG, "Time taken for gpu parallel summation: " + (System.currentTimeMillis() - start));
		tv1.append("\nSum: " + sum); */
		
		/* double[][][][] tearr = new double[][][][] {{{{1.0,2.0,3.0},{4.0,5.0,6.0}}, {{7.0,8.0,9.0},{10.0,11.0,12.0}}}, {{{13.0,14.0,15.0},{16.0,17.0,18.0}}, {{19.0,20.0,21.0},{22.0,23.0,24.0}}}};
		float[] tematarr = new float[] {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f,16.0f,17.0f,18.0f,19.0f,20.0f,21.0f,22.0f,23.0f,24.0f};
		MatrixParallel tesor = new MatrixParallel(Arrays.asList(2,2,2,3), this);
		tesor.tarr = tematarr.clone();
		Log.d(TAG, "Shape 0: " + tearr.length + " Shape 1: " + tearr[0].length + " Shape 2: " + tearr[0][0].length + " Shape 3: " + tearr[0][0][0].length);
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					for (int l = 0; l < 3; l++) {
						Log.d(TAG, "Array value: " + tearr[i][j][k][l] + " Matrix value: " + tesor.get(Arrays.asList(i,j,k,l)));
					}
				}
			}
		} */
		
		
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// Handle action bar item clicks here. The action bar will
		// automatically handle clicks on the Home/Up button, so long
		// as you specify a parent activity in AndroidManifest.xml.
		int id = item.getItemId();
		/* if (id == R.id.action_settings) {
			return true;
		} */
		return super.onOptionsItemSelected(item);
	}
	
	
}

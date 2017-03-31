package en.menghui.android.linearalgebraparallel.maths;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import en.menghui.android.linearalgebraparallel.utils.ArraysUtils;
import android.annotation.SuppressLint;
import android.content.Context;
import android.util.Log;
import Jama.Matrix;

public class MatrixParallel {
	private static final String TAG = "Matrix Parallel";
	
	private Context context;
	
	public Matrix tmat;
	public float[] tarr;
	public List<Integer> shape = new ArrayList<Integer>();
	
	static {
        System.loadLibrary("JNIProcessor");
    }
	
	native private boolean compileKernels();
	native private float[] addVector(float[] inA, float[] inB, float[] out, int width, int height);
	native private float[] addScalarVector(float[] inA, float inB, float[] out, int width, int height);
	native private float[] subtractVector(float[] inA, float[] inB, float[] out, int width, int height);
	native private float[] subtractScalarVector(float[] inA, float inB, float[] out, int width, int height);
	native private float[] scalarSubtractVector(float[] inA, float inB, float[] out, int width, int height);
	native private float[] dotVector(float[] inA, float[] inB, float[] out, int aWidth, int aHeight, int bWidth, int bHeight, int outWidth, int outHeight);
	native private float[] mulVector(float[] inA, float[] inB, float[] out, int width, int height);
	native private float[] mulScalarVector(float[] inA, float inB, float[] out, int width, int height);
	native private float[] divideVector(float[] inA, float[] inB, float[] out, int width, int height);
	native private float[] divideScalarVector(float[] inA, float inB, float[] out, int width, int height);
	native private float[] scalarDivideVector(float[] inA, float inB, float[] out, int width, int height);
	native private float sumVector(float[] inA, int width, int height);
	
	native private float[] dot(float[] inA, float[] inB, float[] out, int aWidth, int aHeight, int bWidth, int bHeight, int outWidth, int outHeight);
	
	/* public MatrixParallel(int row, int col, Context context) {
		this.context = context;
		tmat = new Matrix(row, col);
		
		initOpenCLKernel();
	} 
	
	public MatrixParallel(int row, int col, double val, Context context) {
		this.context = context;
		tmat = new Matrix(row, col, val);
		
		initOpenCLKernel();
	} */
	
	public MatrixParallel(List<Integer> shape, Context context) {
		this.context = context;
		this.shape = shape;
		
		// this.tmat = new Matrix(shape.get(0), getColumns());
		this.tarr = new float[getSize()];
		
		initOpenCLKernel();
	}
	
	public MatrixParallel(List<Integer> shape, double value, Context context) {
		this.context = context;
		this.shape = shape;
		
		// this.tmat = new Matrix(shape.get(0), getColumns(), value);
		this.tarr = new float[getSize()];
		
		for (int i = 0; i < getSize(); i++) {
			this.tarr[i] = (float)value;
		}
		
		initOpenCLKernel();
	}
	
	public MatrixParallel(List<Integer> shape, boolean random, Context context) {
		this.context = context;
		this.shape = shape;
		
		this.tarr = new float[getSize()];
		
		if (random) {
			for (int i = 0; i < getSize(); i++) {
				this.tarr[i] = (float)Math.random();
			}
			// this.tmat = Matrix.random(shape.get(0), getColumns());
		} else { 
			// this.tmat = new Matrix(shape.get(0), getColumns());
		}
		
		initOpenCLKernel();
	}
	
	private int getSize() {
		int size = 1;
		
		for (int i = 0; i < this.shape.size(); i++) {
			size *= this.shape.get(i);
		}
		
		return size;
	}
	
	private int getColumns() {
		int columns = 1;
		if (this.shape.size() > 1) {
			for (int i = 1; i < this.shape.size(); i++) {
				columns *= this.shape.get(i);
		    }
		}
		
		return columns;
	}
	
	public double get(int n, int d, int y, int x) {
		// int ix = ((this.shape.get(2) * d) + y) * this.shape.get(3) + x;
		// return this.tmat.get(n, ix);
		
		return this.tarr[getIndex(Arrays.asList(n, d, y, x))];
	}
	
	public double get(List<Integer> indexes) {
		/* int ix = 0;
		
		if (indexes.size() > 1) {
			ix = indexes.get(1);
			
			if (indexes.size() > 2) {
				// ix = (this.shape.get(2) * ix) + indexes.get(2);
			    for (int i = 2; i < indexes.size(); i++) {
			    	ix *= this.shape.get(i);
			    	ix += indexes.get(i);
			    }
			}
		
		}
		
		return this.tmat.get(indexes.get(0), ix); */
		
		return this.tarr[getIndex(indexes)];
	}
	
	public void set(int n, int d, int y, int x, double v) {
		// int ix = ((this.shape.get(2) * d) + y) * this.shape.get(3) + x;
		// this.tmat.set(n, ix, v);
		
		this.tarr[getIndex(Arrays.asList(n, d, y, x))] = (float)v;
	}
	
	public void set(List<Integer> indexes, double v) {
		/* int ix = 0;
		
		if (indexes.size() > 1) {
			ix = indexes.get(1);
			
			if (indexes.size() > 2) {
			    for (int i = 2; i < indexes.size(); i++) {
			    	ix *= this.shape.get(i);
			    	ix += indexes.get(i);
			    }
			}
		
		}
		
		this.tmat.set(indexes.get(0), ix, v); */
		
		this.tarr[getIndex(indexes)] = (float)v;
	}
	
	public int getColumnIndex(List<Integer> indexes) {
		int ix = 0;
		
		if (indexes.size() > 1) {
			ix = indexes.get(1);
			
			if (indexes.size() > 2) {
			    for (int i = 2; i < indexes.size(); i++) {
			    	ix *= this.shape.get(i);
			    	ix += indexes.get(i);
			    }
			}
		
		}
		
		return ix;
	}
	
	public int getIndex(List<Integer> indexes) {
		int ix = indexes.get(0);
		
		if (indexes.size() > 1) {
			for (int i = 1; i < indexes.size(); i++) {
		    	ix *= this.shape.get(i);
		    	ix += indexes.get(i);
		    }
		}
		
		return ix;
	}
	
	public void add(int n, int d, int y, int x, double v) {
		int ix = ((this.shape.get(2) * d) + y) * this.shape.get(3) + x;
		
		this.tmat.set(n, ix, this.tmat.get(n, ix) + v);
	}
	
	public MatrixParallel cloneAndZero() {
		return new MatrixParallel(this.shape, 0.0, this.context);
	}
	
	public MatrixParallel clone() {
		MatrixParallel tensor = new MatrixParallel(this.shape, 0.0, this.context);
		int r = this.tmat.getRowDimension();
		int c = this.tmat.getColumnDimension();
		
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				tensor.tmat.set(i, j, this.tmat.get(i, j));
			}
		}
		
		return tensor;
	}
	
	public void setConst(double c) {
		for (int i = 0; i < this.tmat.getRowDimension(); i++) {
			for (int j = 0; j < this.tmat.getColumnDimension(); j++) {
				this.tmat.set(i, j, this.tmat.get(i, j) + c);
			}
		}
	}
	
	@SuppressLint("UseSparseArrays")
	public static MatrixParallel sumTensorAxises(MatrixParallel tensor, List<Integer> axis, Context context) {
		List<Integer> reshape = new ArrayList<Integer>();
		List<Integer> shape = new ArrayList<Integer>();
		
		Map<Integer, Integer> reshapeMap = new HashMap<Integer, Integer>();
		Map<Integer, Integer> shapeMap = new HashMap<Integer, Integer>();
		
		int reshapeCount = 0;
		int shapeCount = 0;
		for (int i = 0; i < tensor.shape.size(); i++) {
			if (axis.indexOf(i) == -1) {
				reshape.add(tensor.shape.get(i));
				reshapeMap.put(i, reshapeCount);
				reshapeCount++;
			} else {
				shape.add(tensor.shape.get(i));
				shapeMap.put(i, shapeCount);
				shapeCount++;
			}
		}
		
		MatrixParallel res = new MatrixParallel(reshape, context);
		
		int reshapeSum = 1;
		for (int i = 0; i < reshape.size(); i++) {
			reshapeSum *= reshape.get(i);
		}
		
		int shapeSum = 1;
		for (int i = 0; i < shape.size(); i++) {
			shapeSum *= shape.get(i);
		}
		
		List<Integer> reshape2 = new ArrayList<Integer>();
		List<Integer> shape2 = new ArrayList<Integer>();
		for (int a = 0; a < reshapeSum; a++) {
			reshape2 = new ArrayList<Integer>();
			int reshapeSize = 1;
			for (int b = 0; b < reshape.size(); b++) {
				reshapeSize *= reshape.get(b);
				
				reshape2.add((a / (reshapeSize / reshape.get(b))) % reshape.get(b));
			}
			
			Collections.reverse(reshape2);
			// Log.d(TAG, "Reshape 2: " + reshape2.toString());
			
			
			double sum = 0.0;
			for (int c = 0; c < shapeSum; c++) {
				shape2 = new ArrayList<Integer>();
				int shapeSize = 1;
			    for (int d = 0; d < shape.size(); d++) {
			    	shapeSize *= shape.get(d);
			    	
					shape2.add((c / (shapeSize / shape.get(d))) % shape.get(d));
			    }
			    
			    Collections.reverse(shape2);
			    // Log.d(TAG, "Shape 2: " + shape2.toString());
			    
			    List<Integer> list = new ArrayList<Integer>();
			    for (int e = 0; e < tensor.shape.size(); e++) {
			    	if (reshapeMap.containsKey(e)) {
			    		list.add(reshape2.get(reshapeMap.get(e)));
			    	} else {
			    		list.add(shape2.get(shapeMap.get(e)));
			    	}
			    }
			    
			    // Log.d(TAG, "Shape List: " + list.toString());
			    sum += tensor.get(list);
				
		    }
			
			res.set(reshape2, sum);
		}
		
		return res;
	}
	
	public MatrixParallel add(MatrixParallel b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = addVector(this.tarr, b.tarr, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel add(double b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = addScalarVector(this.tarr, (float)b, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel subtract(MatrixParallel b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = subtractVector(this.tarr, b.tarr, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel subtract(double b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = subtractScalarVector(this.tarr, (float)b, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel subtractFrom(double b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = scalarSubtractVector(this.tarr, (float)b, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel dot(MatrixParallel b) {
		MatrixParallel out = new MatrixParallel(Arrays.asList(this.shape.get(0), b.shape.get(1)), this.context);
		
		out.tarr = dotVector(
				this.tarr, 
				b.tarr, 
				out.tarr, 
				this.shape.get(1), 
				this.shape.get(0), 
				b.shape.get(1), 
				b.shape.get(0), 
				out.shape.get(1), 
				out.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel mul(MatrixParallel b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = mulVector(this.tarr, b.tarr, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel mul(double b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = mulScalarVector(this.tarr, (float)b, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel divide(MatrixParallel b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = divideVector(this.tarr, b.tarr, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel divide(double b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = divideScalarVector(this.tarr, (float)b, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public MatrixParallel divideFrom(double b) {
		MatrixParallel out = new MatrixParallel(this.shape, this.context);
		
		out.tarr = scalarDivideVector(this.tarr, (float)b, out.tarr, this.shape.get(1), this.shape.get(0));
		
		return out;
	}
	
	public float sum() {
		float sum = sumVector(this.tarr, this.shape.get(1), this.shape.get(0));
		
		return sum;
	}
	
	private void initOpenCLKernel() {
		copyFile("kernels.cl");
		
        if( compileKernels() == false ) {
            Log.i(TAG, "Kernel compilation failed");
        } else {
        	Log.i(TAG, "Kernel compilation passed");
        }
	}
	
	private void copyFile(final String f) {
		InputStream in;
		try {
			in = this.context.getAssets().open(f);
			final File of = new File(this.context.getDir("execdir", Context.MODE_PRIVATE), f);
			
			final OutputStream out = new FileOutputStream(of);

			final byte b[] = new byte[65535];
			int sz = 0;
			while ((sz = in.read(b)) > 0) {
				out.write(b, 0, sz);
			}
			in.close();
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

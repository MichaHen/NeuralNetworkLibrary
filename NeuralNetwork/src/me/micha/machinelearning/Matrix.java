package me.micha.machinelearning;

public class Matrix {

	public double[][] matrix;
	public int rows;
	public int columns;
	
	public Matrix(int rows, int columns) {
		this(rows, columns, 0);
	}
	
	public Matrix(int rows, int columns, double a) {
		matrix = new double[rows][columns];
		this.rows = rows;
		this.columns = columns;
		
		init(a);
	}
	
	public Matrix(double[][] matrix) {
		this.matrix = matrix;
		rows = matrix.length;
		columns = matrix[0].length;
	}
	
	//OBJECT
	
	public void init(double a) {
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] = a;
			}
		}
	}
	
	public Matrix clone() {
		Matrix n =  new Matrix(rows, columns);
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				n.matrix[i][j] = matrix[i][j];
			}
		}
		
		return n;
	}
	
	public double[][] getMatrix() {
		return matrix;
	}
	
	public double[] toArray() {
		double[] array = new double[rows * columns];
		int c = 0;
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				array[c] = (matrix[i][j]);
				c++;
			}
		}
		
		return array;
	}
	
	public String toString() {
		StringBuilder b = new StringBuilder("[");
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				b.append(matrix[i][j] + ", ");
			}
			b.setLength(b.length()-2);
			b.append("]");
			if((i+1) < rows) {
				b.append(System.lineSeparator());
				b.append("[");
			}
		}
		
		return b.toString();
	}
	
	public boolean equals(Matrix m) {
		if(!(rows == m.rows && columns == m.columns)) return false;
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				if(m.matrix[i][j] != matrix[i][j]) return false; 
			}
		}
		return true;
	}
	
	public void randomize() {
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] = Math.random() * 2 - 1;
			}
		}
	}
	
	//MATH
	
	public void add(double a) {
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] += a;
			}
		}
	}
	
	public void add(Matrix m) {
		if(!(m.rows == rows && m.columns == columns)) return;
		
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] += m.getMatrix()[i][j];
			}
		}
	}
	
	public void subtract(double a) {
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] -= a;
			}
		}
	}
	
	public void subtract(Matrix m) {
		if(!(m.rows == rows && m.columns == columns)) return;
		
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] -= m.getMatrix()[i][j];
			}
		}
	}
	
	public void multiply(double a) {
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] *= a;
			}
		}
	}
	
	public void multiply(Matrix m) {
		if(!(m.rows == rows && m.columns == columns)) return;
		
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] *= m.getMatrix()[i][j];
			}
		}
	}
	
	public void divide(double a) {
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] /= a;
			}
		}
	}
	
	public void divide(Matrix m) {
		if(!(m.rows == rows && m.columns == columns)) return;
		
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				matrix[i][j] /= m.getMatrix()[i][j];
			}
		}
	}
	
	public double maxArg(int row) {
		double h = 0;
		
		for(int i = 0; i < this.columns; i++) {
			if(matrix[row][i] > h) {
				h = matrix[row][i];
			}
		}
		
		return h;
	}
	
	public int maxArgCol(int col) {
		double h = 0;
		int index = 0;
		
		for(int i = 0; i < this.rows; i++) {
			if(matrix[i][col] > h) {
				h = matrix[i][col];
				index = i;
			}
		}
		
		return index;
	}
	
	//STATIC
	
	public static Matrix matrixProduct(Matrix a, Matrix b) {
		if((a.columns != b.rows)) return null;

		
		Matrix matrix = new Matrix(a.rows, b.columns);
		
		
		for(int i = 0; i < matrix.rows; i++) {
			for(int j = 0; j < matrix.columns; j++) {
				double sum = 0;
				
				for(int k = 0; k < a.columns; k++) {
					sum += a.matrix[i][k] * b.getMatrix()[k][j];
				}
				
				matrix.getMatrix()[i][j] = sum;
				
			}
		}
		
		return matrix;
	}
	
	public static Matrix transpose(Matrix a) {
		Matrix result = new Matrix(a.columns, a.rows);
		
		for(int i = 0; i < a.rows; i++) {
			for(int j = 0; j < a.columns; j++) {
				result.matrix[j][i] = a.matrix[i][j];
			}
		}
		
		return result;
	}
	
	public static Matrix fromArray(double[] array) {
		Matrix matrix = new Matrix(array.length, 1);
		
		for(int i = 0; i < array.length; i++) {
			matrix.matrix[i][0] = array[i];
		}
		
		return matrix;
	}
	
}

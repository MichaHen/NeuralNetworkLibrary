package me.micha.machinelearning.lib;

public class Matrix {
	
	//Matrizen sind vektorisiert implementiert, da eindimensionale Arrays die Rechenzeit erheblich verringern.
	public double[] matrix;
	//Anzahl Reihen
	public int rows;
	//Anzahl Spalten
	public int columns;
	
	public Matrix(int rows, int columns) {
		this(rows, columns, 0);
	}
	
	public Matrix(int rows, int columns, double a) {
		matrix = new double[rows*columns];
		this.rows = rows;
		this.columns = columns;
		
		init(a);
	}
	
	public Matrix(double[][] matrix) {
		rows = matrix.length;
		columns = matrix[0].length;
		this.matrix = new double[rows*columns];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				s(i, j, matrix[i][j]);
			}
		}
	}
	
	//Matrix von Array
	public Matrix(double[] matrix, int rows, int columns) {
		this.rows = rows;
		this.columns = columns;
		this.matrix = new double[rows*columns];
		
		//Überschreiben der Werte, damit der Originalarray nicht verändert wird
		for(int i = 0; i < matrix.length; i++) {
			this.matrix[i] = matrix[i];
		}
	}
	
	//OBJECT
	
	public double g(int i, int j) {
		if(i >= rows || j >= columns) throw new IndexOutOfBoundsException();
		return matrix[columns * i + j];
	}
	
	public void s(int i, int j, double d) {
		if(i >= rows || j >= columns) return;
		matrix[columns * i + j] = d;
	}
	
	//Referenzierung in vektorisierter Matrizenform
	public int indexFetch(int i, int j) {
		return (i * columns + j);
	}
	
	//Fuellt Matrix mit Wert a
	public void init(double a) {
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] = a;
		}
	}
	
	//Klont Matrix. Dabei wird ein neuer Array erstellt, damit Überschneidungen durch die Präsenz des selben Arrays in versch. Matrix-Objekten vermieden werden.
	public Matrix clone() {
		Matrix n =  new Matrix(rows, columns);
		for(int i = 0; i < matrix.length; i++) {
			n.matrix[i] = matrix[i];
 		}
		
		return n;
	}
	
	
	public double[] getMatrix() {
		return matrix;
	}
	
//	public double[] toArray() {
//		double[] array = new double[rows*columns];
//		int c = 0;
//		for(int i = 0; i < rows; i++) {
//			for(int j = 0; j < columns; j++) {
//				array[c] = matrix[indexFetch(i, j)];
//				c++;
//			}
//		}
//		
//		return array;
//	}
	
	//String-Formatierung zu Debugzwecken
	public String toString() {
		StringBuilder b = new StringBuilder("[");
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < columns; j++) {
				b.append(matrix[indexFetch(i, j)] + ", ");
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
	
	//Gleichheit der Werte von zwei Matrizen zu Debugzwecken
	public boolean equals(Matrix m) {
		if(!(rows == m.rows && columns == m.columns)) return false;
		for(int i = 0; i < matrix.length; i++) {
			if(m.matrix[i] != matrix[i]) return false; 
		}
		return true;
	}
	
	//Zuweisung von Zufallszahlen aus [-1,1]
	public void randomize() {
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] = Global.RAND.nextDouble() * 2 - 1;
		}
	}
	
	//Abschnitt: Mathematische Operationen
	
	//Einfache Addition
	public void add(double a) {
			for(int i = 0; i < matrix.length; i++) {
				matrix[i] += a;
			}
		
	}
	
	//Komponentenweise Addition
	public void add(Matrix m) {
		if(!(m.rows == rows && m.columns == columns)) return;
		
			for(int i = 0; i < matrix.length; i++) {
				matrix[i] += m.matrix[i];
			}
	}
	
	//Einfache Subtraktion
	public void subtract(double a) {
			for(int i = 0; i < matrix.length; i++) {
				matrix[i] -= a;
			}
	}
	
	
	//Komponentenweise Subtraktion
	public void subtract(Matrix m) {
		if(!(m.rows == rows && m.columns == columns)) return;
		
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] -= m.matrix[i];
		}
		
	}
	
	//Skalarprodukt
	public void multiply(double a) {
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] *= a;
		}
	}
	
	//Komponentenweise Multiplikation
	public void multiply(Matrix m) {
		if(!(m.rows == rows && m.columns == columns)) return;
		
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] *= m.matrix[i];
		}
	}
	
	
	//Einfache Division (/Skalarprodukt)
	public void divide(double a) {
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] /= a;
		}
	}
	
	
	//Komponentenweise Division
	public void divide(Matrix m) {
		if(!(m.rows == rows && m.columns == columns)) return;
		
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] /= m.matrix[i];
		}
	}
	
	//Ausgabe des groessten Komponenten in einer Reihe
	public double maxArg(int row) {
		double h = 0;
		
		for(int i = 0; i < this.columns; i++) {
			if(matrix[indexFetch(row, i)] > h) {
				h = matrix[indexFetch(row, i)];
			}
		}
		
		return h;
	}
	
	//Ausgabe des Indexes des groessten Komponenten in einer Spalte
	//Einfache Assoziatin mit einer Ziffer. Ziffer = Index (in Output-Layer)
	public int maxArgCol(int col) {
		double h = 0;
		int index = 0;
		
		for(int i = 0; i < this.rows; i++) {
			if(matrix[indexFetch(i, col)] > h) {
				h = matrix[indexFetch(i, col)];
				index = i;
			}
		}
		
		return index;
	}
	
	
	//Quadrierung der Matrix
	public void squared() {
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] *= matrix[i];
		}
	}
	
	//Quadratwurzel der Matrix
	public void sqrt() {
		for(int i = 0; i < matrix.length; i++) {
			matrix[i] = Math.sqrt(matrix[i]);
		}
	}
	
	//Abschnitt: Statische Methoden
	
	//Matrizen-Produkt
	//Potenziell noch vektorisierte Implementierung
	public static Matrix matrixProduct(Matrix a, Matrix b) {
		if((a.columns != b.rows)) return null;

		
		Matrix matrix = new Matrix(a.rows, b.columns);
		
		
		for(int i = 0; i < matrix.rows; i++) {
			for(int j = 0; j < matrix.columns; j++) {
				double sum = 0;
				
				for(int k = 0; k < a.columns; k++) {
					sum += a.matrix[a.indexFetch(i, k)] * b.matrix[b.indexFetch(k, j)];
				}
				
				matrix.matrix[matrix.indexFetch(i, j)] = sum;
				
			}
		}
		
		return matrix;
	}
	
//	public static Matrix tensorProduct(Matrix a, Matrix b) {
//		if((a.columns != b.rows)) return null;
//		
//		Matrix matrix = new Matrix(a.rows, b.columns);
//		
//		for(int i = 0; i < matrix.rows; i++) {
//			for(int j = 0; j < matrix.columns; j++) {
//				matrix.matrix[matrix.indexFetch(i, j)] = a.matrix[a.indexFetch(i, 0)] * b.matrix[b.indexFetch(0, j)];
//			}
//		}
//		
//		return matrix;
//		
//	}
	
	//Transponieren der Matrix
	//=Vertauschen der Indizes
	public static Matrix transpose(Matrix a) {
		Matrix result = new Matrix(a.columns, a.rows);
		
		for(int i = 0; i < a.rows; i++) {
			for(int j = 0; j < a.columns; j++) {
				result.matrix[result.indexFetch(j, i)] = a.matrix[a.indexFetch(i, j)];
			}
		}
		
		return result;
	}
	
	//Statische Array-Parse-Methode
	public static Matrix fromArray(double[] array) {
		Matrix matrix = new Matrix(array.length, 1);
		
		for(int i = 0; i < array.length; i++) {
			matrix.matrix[i] = array[i];
		}
		
		return matrix;
	}
	
}

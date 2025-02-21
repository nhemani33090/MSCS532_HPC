# **🔢 Optimized Matrix Multiplication in Python**

## **📖 Overview**  
This repository contains implementations of **matrix multiplication**, showcasing **three different approaches** with performance comparisons. The goal is to demonstrate how **data locality optimization** and **parallel computing** improve execution efficiency.

### **1️⃣ Implemented Approaches**
- **Basic Matrix Multiplication** - A simple triple-nested loop implementation.
- **NumPy Optimized Implementation** - Uses `numpy.dot()` for highly efficient, vectorized computation.
- **Parallelized Matrix Multiplication** - Divides the computation across multiple CPU cores using Python's `multiprocessing` module.

---

## **🛠 Requirements**  
- **Python 3.x**  
- **NumPy**  
- **Matplotlib** (for optional visualization)  

Install dependencies using:  
```bash
pip install numpy matplotlib
```

---

## **📌 How to Run the Code**  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/nhemani33090/MSCS532_HPC
cd MSCS532_HPC
```

### **2️⃣ Run the Script**
```bash
python3 matrix_multiplication.py
```

This will execute all three implementations and display performance comparisons.

---

## **📊 Performance Summary**
### **Execution Times (200x200 Matrix)**
| Implementation      | Execution Time (seconds) |
|--------------------|------------------------|
| Basic             | **3.894164**           |
| NumPy Optimized   | **0.001899**           |
| Parallelized      | **0.038939**           |

### **Final Speed Comparisons**
- **NumPy Optimized vs. Basic:** **~2052x faster**
- **Parallelized vs. Basic:** **~99.9x faster**
- **NumPy Optimized vs. Parallelized:** **~20.5x faster**

### **Key Takeaways**
✅ **NumPy is the fastest approach** due to efficient memory usage and vectorized operations.  
✅ **Parallelization improves performance** but has overhead from process management.  
✅ **Basic implementation is the slowest** due to inefficient cache usage.  

---

## **📜 Lessons Learned**
- **Data locality optimization significantly improves performance.**  
- **Vectorized operations (NumPy) outperform explicit loops.**  
- **Parallelization helps but is not always better than optimized single-threaded execution.**  

📌 **For large-scale computations, NumPy and hardware-aware parallelization techniques are preferred.**  

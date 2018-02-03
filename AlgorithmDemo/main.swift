//
//  main.swift
//  AlgorithmDemo
//
//  Created by Cobb on 2017/7/26.
//  Copyright © 2017年 Cobb. All rights reserved.
//

import Foundation

print("Hello, World!")

//插入排序
var cards = [5, 7, 18, 12, 49, 30, 3, 19, 99, 14]
for j in 1..<cards.count {
    let key = cards[j]
    var i = j - 1
    while i >= 0 && cards[i] > key {
        cards[i + 1] = cards[i]
        i = i - 1
    }
    cards[i + 1] = key
}
print(cards)

//归并排序 (A[p, q],A[q + 1, r]都已排好序)
func merge(A:inout Array<Int>, p:Int, q:Int, r:Int) {
    let n1 = q - p
    let n2 = r - q - 1
    var L:Array<Int> = []
    var R:Array<Int> = []
    for i in 0...n1 {
        L.append(A[p + i])
    }
    L.append(NSIntegerMax)
    for j in 0...n2 {
        R.append(A[q + j + 1])
    }
    R.append(NSIntegerMax)
    var i = 0
    var j = 0
    for k in p...r {
        if L[i] <= R[j] {
            A[k] = L[i]
            i = i + 1
        } else {
            A[k] = R[j]
            j = j + 1
        }
    }
}

//分治法 递归调用 (尚未考虑数组count为奇数情况)
func mergeSort(A:inout Array<Int>, n:Int) {
    if n * 2 >  A.count {
        return
    }
    let mergeCount = ((A.count % 2) == 0) ?  (A.count / (2 * n)) : ((A.count + 1) / (2 * n))
    for i in 0..<mergeCount {
        merge(A: &A, p: (i * n * 2), q: ((i * 2 + 1) * n - 1), r: ((i + 1) * n * 2 - 1))
    }
    mergeSort(A: &A, n: n * 2)
}

var A = [12, 14, 25, 47, 1, 16, 23, 36]
merge(A: &A, p: 0, q: 3, r: A.count - 1)
print(A)

var B = [12, 14, 45, 47, 31, 16, 93, 36]
mergeSort(A: &B, n: 1)
print(B)


//分治策略
//最大子数组使用分治策略的求解方法
//获取跨越中点的最大子数组的边界
func findMaxCrossingSubArray(A:inout Array<Int>, low:Int, mid:Int, high:Int) -> (maxLeft:Int, maxRight:Int, sum:Int) {
    var leftSum = -NSIntegerMax
    var rightSum = -NSIntegerMax
    var maxLeft = low
    var maxRight = high
    var sum = 0
    for i in low...mid {
        sum = sum + A[mid - (i - low)]
        if sum > leftSum {
            leftSum = sum
            maxLeft = mid - (i - low)
        }
    }
    sum = 0
    for j in mid + 1...high {
        sum = sum + A[j]
        if sum > rightSum {
            rightSum = sum
            maxRight = j
        }
    }
    return (maxLeft, maxRight, leftSum + rightSum);
}

var crossingSharePrices = [-80, 12, -4, 25, 47, 1, -16, 23, -36, -99, 34, 23]
let crossingSubArray = findMaxCrossingSubArray(A: &crossingSharePrices, low: 0, mid: 6, high: 11)
print(crossingSubArray)

//使用分治策略求解最大子数组问题
func findMaximumSubArray(A:inout Array<Int>, low:Int, high:Int) -> (low:Int, high:Int, sum:Int) {
    if high == low {
        return (low, high, A[low])
    } else {
        let mid = (low + high) / 2
        let (leftLow, leftHigh, leftSum) = findMaximumSubArray(A: &A, low: low, high: mid)
        let (rightLow, rightHigh, rightSum) = findMaximumSubArray(A: &A, low: mid + 1, high: high)
        let (crossLow, crossHigh, crossSum) = findMaxCrossingSubArray(A: &A, low: low, mid: mid, high: high)
        if leftSum > rightSum && leftSum > crossSum {
            return (leftLow, leftHigh, leftSum)
        } else if rightSum > leftSum && rightSum > crossSum {
            return (rightLow, rightHigh, rightSum)
        } else {
            return (crossLow, crossHigh, crossSum)
        }
    }
}

var sharePrices = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
let result = findMaximumSubArray(A: &sharePrices, low: 0, high: sharePrices.count - 1)
print(result)

//矩阵乘法Ω（n³）
func squareMatrixMultiply(A: [[Int]], B: [[Int]]) -> [[Int]] {
    let n = A.count
    var C = [[Int]]()
    for _ in 0..<n {
        var b = [Int]()
        for _ in 0..<n {
            b.append(0)
        }
        C.append(b)
    }
    for i in 0..<n {
        for j in 0..<n {
            C[i][j] = 0
            for k in 0..<n {
                C[i][j] += A[i][k] * B[k][j]
            }
        }
    }
    return C
}

let matrixA = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
let matrixB = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
let matrixC = squareMatrixMultiply(A: matrixA, B: matrixB)
print(matrixC)

//矩阵乘法的strassen算法 运行时间为Θ(n的lg7次方) Ο(n的2.81次方)
/**
 func squareMatrixMultiplyRecursive(A: [[Int]], B: [[Int]]) -> [[Int]] {
 let n = A.count
 var C = [[Int]]()
 for _ in 0..<n {
 var b = [Int]()
 for _ in 0..<n {
 b.append(0)
 }
 C.append(b)
 }
 if n == 1 {
 C[1][1] = A[1][1] * B[1][1]
 } else {
 C[1][1] = squareMatrixMultiplyRecursive(A: A[1][1], B: B[1][1]) +
 squareMatrixMultiplyRecursive(A: A[1][2], B: B[2][1])
 C[1][2] = squareMatrixMultiplyRecursive(A: A[1][1], B: B[1][2]) +
 squareMatrixMultiplyRecursive(A: A[1][2], B: B[2][2])
 C[2][1] = squareMatrixMultiplyRecursive(A: A[2][1], B: B[1][1]) +
 squareMatrixMultiplyRecursive(A: A[2][2], B: B[2][1])
 C[2][2] = squareMatrixMultiplyRecursive(A: A[2][1], B: B[1][2]) +
 squareMatrixMultiplyRecursive(A: A[2][2], B: B[2][2])
 }
 return C;
 }
 */

//最大堆
func maxHeapify(A:inout [Int], index:Int) {
    let l = (index + 1)<<1 - 1
    let r = (index + 1)<<1
    var largest = index
    if l < A.count && A[l] > A[index] {
        largest = l
    }
    if r < A.count && A[r] > A[largest] {
        largest = r
    }
    if largest != index {
        let temp = A[largest]
        A[largest] = A[index]
        A[index] = temp
        maxHeapify(A: &A, index: largest)
    }
}

//建立最大堆
func buildMaxHeap(A:inout [Int]) {
    for index in 1...A.count / 2 {
        maxHeapify(A: &A, index: A.count / 2 - index)
    }
}

var heapA = [16, 4, 10, 14, 7, 9, 3, 2, 8, 1];
print(heapA)
buildMaxHeap(A: &heapA)
print(heapA)

//快速排序

/// 数组划分，找到小标q A[p..r]为A[p..q-1]和A[q+1..r],使得A[p..q-1]中的每一个元素小于等于A[q],而A[q]小于等于A[q+1..r]中的每个元素
///
/// - Parameter A: 排序的数组
/// - Parameter p: 排序开始位置
/// - Parameter r: 排序结束位置
/// - Returns: 返回数组划分的小标q
func partition(A:inout [Int], p: Int, r: Int) -> Int {
    let x = A[r]
    var i = p
    for j in p..<r {
        if A[j] <= x {
            i += 1
            // 第i个元素和第j个元素交换
            let temp = A[j]
            A[j] = A[i - 1]
            A[i - 1] = temp
        }
    }
    // 第i + 1个元素和最后一个元素交换
    let temp = A[i]
    A[i] = A[r]
    A[r] = temp
    return i
}

/// 递归调用快速排序
///
/// - Parameter A: 排序数组
func quickSort(A:inout [Int], p: Int, r: Int) {
    if p < r {
        let q = partition(A: &A, p: p, r: r)
        quickSort(A: &A, p: p, r: q - 1)
        quickSort(A: &A, p: q + 1, r: r)
    }
    print("quickSort current change \(A)")
}

var quickSortA = [2, 8, 7, 1, 3, 5, 6, 4, 80, 11, 29]
print(quickSortA)
quickSort(A: &quickSortA, p: 0, r: quickSortA.count - 1)
print(quickSortA)

//线性时间排序
//记数排序 (基数排序)

func countingSort(A:inout [Int], B:inout [Int], k: Int) {
    var C: [Int] = []
    for _ in 0...k {
        C.append(0)
    }
    for j in 0..<A.count {
        C[A[j]] = C[A[j]] + 1
    }
    // C[i] now contains the number of elements equal to i.
    for i in 1...k {
        C[i] = C[i] + C[i - 1]
    }
    // C[i] now contains the number of elements less than or equal to i.
    for j in 1...A.count {
        let revert = A.count - j
        B[C[A[revert]]] = A[revert]
        C[A[revert]] = C[A[revert]] - 1
    }
}

var countingSortA = [2, 5, 3, 0, 2, 3, 0, 3]
var outPutB: [Int] = []
for _ in 0...countingSortA.count {
    outPutB.append(0)
}
countingSort(A: &countingSortA, B: &outPutB, k: 5)
print(outPutB)

// 桶排序
func bucketSort(A:inout [Float]) {
    let n = A.count
    var B: [Float] = []
    for _ in 0..<n {
        B.append(0)
    }
    for i in 1...n {
//        inset A[i] into list B[「n * A[i]」]
    }
    for i in 0..<n {
//        sort list B[i] with insertion sort
    }
//    contatenate the lists B[0], B[1],...,B[n - 1] together in order
}

// 中位数和顺序统计量
/// 找到数组A【p..r】中第i小的元素
func randomizedSelect(A:inout [Int], p: Int, r: Int, i: Int) -> Int {
    if p == r {
        return A[p]
    }
    let q = partition(A: &A, p: p, r: r)
    let k = q - p + 1
    if i == k {
        return A[q]
    } else if i < k {
        return randomizedSelect(A: &A, p: p, r: q - 1 , i: i)
    } else {
        return randomizedSelect(A: &A, p: q + 1, r: r , i: i - k)
    }
}

// 数据结构
// 基本数据结构 栈和队列 132 - 238









































































































































































































































































































































# coding=utf-8
def quick_sort(array, start, end):
    if start >= end:
        return
    mid_data, left, right = array[start], start, end
    while left < right:
        while array[right] >= mid_data and left < right:
            right -= 1
        array[left] = array[right]
        while array[left] < mid_data and left < right:
            left += 1
        array[right] = array[left]
    array[left] = mid_data
    quick_sort(array, start, left - 1)
    quick_sort(array, left + 1, end)
def quickSoft(array,start,end):
    if start>=end:
        return
    mid_data,left,right = array[start],start,end
    while left < right:
        while array[right] >= mid_data and left < right:
            right -= 1
        array[left] = array[right]
        while array[left]<mid_data and left < right:
            left +=1
        array[right] = array[left]
    array[left] = mid_data
    quickSoft(array,start,left-1)
    quickSoft(array,left+1,end)
def qs(array,start,end):
    if start>end:
        return
    mid_data ,left,right = array[start],start,end
    while(left<right):
        while(mid_data<=array[right] && left<right):
            right -= 1
        array[left] = array[right]
        while(mid_data>array[left] && left<right):
            left +=1
        array[right] = array[left]

        array[left] = mid_data
        qs(array,start,left-1)
        qs(array,left+1,end)
    def qs2(array,start,end):
        if start>end:
            return
        mid_data,left,right = array[start],start,end
        while(left<right):
            while(mid_data<array[right] && left<right):
                right -=1
            array[left] = array[right]
            while(mid_data>array[left] && left<right):
                left +=1
            array[right] = array[left]

            array[left] = mid_data
            qs(array,start,left-1)
            qs(array,left+1,end)

if __name__ == '__main__':
    # array = [10, 17, 50, 7, 30, 24, 27, 45, 15, 5, 36, 21]
    # quick_sort(array, 0, len(array) - 1)
    # print(array)
    array=[]
    quickSoft(array,0,len(array)-1)
    print(array)


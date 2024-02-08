def sortBubbleAlgo(to_be_sorted):
    n = len(to_be_sorted)
    for i in range(n):
        for j in range(0, n - i - 1):
            if to_be_sorted[j] > to_be_sorted[j + 1]:
                to_be_sorted[j], to_be_sorted[j + 1] = to_be_sorted[j + 1], to_be_sorted[j]

if __name__ == "__main__":
  to_be_sorted = [ 3, 1, 16, 28 ]
 
  sortBubbleAlgo(to_be_sorted)
 
  print("Sorted array is:")
  for i in range(len(to_be_sorted)):
      print("%d" % to_be_sorted[i])
  
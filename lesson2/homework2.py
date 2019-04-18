# Homework Week 2:
#
# 1. 【Coding】:
#    Finish 2D convolution/filtering by your self. 
#    What you are supposed to do can be described as "median blur", which means by using a sliding window 
#    on an image, your task is not going to do a normal convolution, but to find the median value within 
#    that crop.
#
#    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When 
#    "REPLICA" are given to you, the padded pixels are the same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis 
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version 
#    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#    Follow up 1: Can it be completed in a shorter time complexity?
#    Follow up 2: Can it be completed in O(W·H·m·n)?
#
   # Python version:


def padding(img, kernel, padding_way = 'ZERO'):

    ker_w = len(kernel[0])
    ker_l = len(kernel)
    num_pad_row = int(ker_w/2)
    num_pad_len = int(ker_l/2)

    if padding_way == 'ZERO':

        for i in range(num_pad_row):
            # padd kernel_l/2 times
            for row in img:
                # padd row head and trail with o
                row.insert(0,0)
                row.append(0)

        for j in range(num_pad_len):
            # padd kernel_w/2 times
            img.append([0]*len(img[0])) # padd head row and trail row with 0
            img.insert(0, [0]*len(img[0]))

    if padding_way == 'REPLICA':

        for i in range(num_pad_row):
            for row in img:
                row.insert(0, row[0]) # padd with first element
                row.append(row[-1])   # padd with last element

        for j in range(num_pad_len):
            img.append(img[-1])     # padd with top row
            img.insert(0,img[0])    # padd with last row

    return img

def padding_np(img, kernel, padding_way = 'ZERO'):
    '''
    if np is allowed, that must be fucking easier
    '''
    import numpy as np
    img = np.array(img)
    kernel = np.array(kernel)
    l, w = kernel.shape
    if padding_way == 'ZERO':
        img = np.pad(img, ((int(l/2), int(l/2)), (int(w/2), int(w/2))), 'constant', constant_values=0)
    if padding_way == 'REPLICA':
        img = np.pad(img, ((int(l/2), int(l/2)), (int(w/2), int(w/2))), 'edge')

    return img.tolist()

def conv(crop_img, kernel):

    import numpy as np

    ker_width = len(kernel[0])
    ker_length = len(kernel)
    conv = []

    # 2D
    for i in range(ker_length):
        for j in range(ker_width):
            conv.append(kernel[i][j]*crop_img[-i-1][-j-1])
            # print(
            #     '{}*{} = {}'.format(kernel[i][j],
            #                         crop_img[-i-1][-j-1],
            #                         kernel[i][j]*crop_img[-i-1][-j-1]),
            #     'conv = {}, median= {}'.format(conv,
            #                                    int(np.median(conv)))
            # )
    conv = int(np.median(conv))
    return conv

def crop_img(x,y, pad_img, ker_l, ker_w):
    import numpy as np
    pad_img = np.array(pad_img)
    # print(type(pad_img))
    # print(
    #     'ker_;{}'.format(ker_l),
    #     'ker_w{}'.format(ker_w)
    # )
    # print(pad_img)
    crop_img = pad_img[x:x+ker_l, y:y+ker_w] # same size as kernel
    return crop_img

def medianBlur(img, kernel, padding_way):
#    # img & kernel is List of List; padding_way a string
#    # Please finish your code under this blank
    new_row = []
    row_l = len(img)
    column_l = len(img[0])

    # padding
    pad_img = padding(img,kernel,padding_way)

    for row in range(row_l):
       temp = []
       for column in range(column_l):
           cropimg = crop_img(row,column,pad_img,len(kernel), len(kernel[0])) # crop kernel size to do conv
           temp.append(conv(cropimg,kernel)) # add new row element by conv
       new_row.append(temp) # add new row
    return new_row

# img = [[1,2,1,3,1],[4,1,3,1,5],[7,1,1,1,9],[2,1,1,1,9],[2,3,5,3,1]]
# kernel = [[1,1,1],[1,1,1],[1,1,1]]
# print(padding(img, kernel,'ZERO'))
# print(crop_img(2,2, pad_img, 3,3))
# img = [[1,2,1,3,1],[4,1,3,1,5],[7,1,1,1,9],[2,1,1,1,9],[2,3,5,3,1]]
# kernel = [[1,0,1],[1,0,1],[1,0,1]]
# pad_img = padding(img, kernel,'REPLICA')
# crop_img = [[1,1,1]]
# kernel = [[1,1,1]]
# print(conv(crop_img,kernel))
# print(medianBlur(img,kernel,'REPLICA'))





#
#//   C++ version:
#//   void medianBlur(vector<vector<int>>& img, vector<vector<int>> kernel, string padding_way){
#//       Please finish your code within this blanck  
#//   }
#
#    We recommend to try in both language (Especially in C++).
#    Good Luck!

#    2. 【Reading + Pseudo Code】
#       We haven't told RANSAC algorithm this week. So please try to do the reading.
#       And now, we can describe it here:
#       We have 2 sets of points, say, Points A and Points B. We use A.1 to denote the first point in A, 
#       B.2 the 2nd point in B and so forth. Ideally, A.1 is corresponding to B.1, ... A.m corresponding 
#       B.m. However, it's obvious that the matching cannot be so perfect and the matching in our real
#       world is like: 
#       A.1-B.13, A.2-B.24, A.3-x (has no matching), x-B.5, A.4-B.24(This is a wrong matching) ...
#       The target of RANSAC is to find out the true matching within this messy.
#       
#       Algorithm for this procedure can be described like this:
#       1. Choose 4 pair of points randomly in our matching points. Those four called "inlier" (中文： 内点) while 
#          others "outlier" (中文： 外点)
#       2. Get the homography of the inliers
#       3. Use this computed homography to test all the other outliers. And seperated them by using a threshold 
#          into two parts:
#          a. new inliers which is satisfied our computed homography
#          b. new outliers which is not satisfied by our computed homography.
#       4. Get our all inliers (new inliers + old inliers) and goto step 2
#       5. As long as there's no changes or we have already repeated step 2-4 k, a number actually can be computed,
#          times, we jump out of the recursion. The final homography matrix will be the one that we want.
#
#       [WARNING!!! RANSAC is a general method. Here we add our matching background to that.]
#
#       Your task: please complete pseudo code (it would be great if you hand in real code!) of this procedure.
#
#       Python:
def ransacMatching(A, B):
  # A & B: List of List
  # A:[[x1,y2],[x2,y2]....]
  # B:[[x1,y2],[x2,y2]....]
  import numpy as np

  K = 10000
  num_inner = 0
  home = {'w':0, 'b':0}
  repeat = 0
  for i in range(K):
      shreshold = 2
      pointA = A[np.asscalar(np.random.randint(0,len(A),1,'int'))]
      pointB = B[np.asscalar(np.random.randint(0,len(B),1,'int'))]
      if (pointA[0]-pointB[0]) == 0:
          w = 'noneexist'
          continue
      else:
          w = (pointA[1]-pointB[1])/(pointA[0]-pointB[0])*1.0
          b = pointA[1] - w * pointA[0]


      listtotal = []
      for point in A+B:
          if point not in listtotal:
              listtotal.append(point)
      inner = 0
      for i in listtotal:
          if abs(i[1]-w*i[0]+b)<=shreshold:
              inner += 1


      if num_inner == inner:
          repeat += 1
      else:
          repeat = 0

      num_inner = max(num_inner, inner)
      if num_inner == inner:
          home['w'] = w
          home['b'] = b
      if repeat >= 10:
          break
  print("homegraphy will be y = {} * x + ({}) with {} inliers".format(home['w'],home['b'],num_inner))
  return home

ransacMatching([[1,1],[2,2],[5,6]],[[3,4],[11,5],[99,1],[33,4],[44,5],[2,3]])



#//     C++:
#//     vector<vector<float>> ransacMatching(vector<vector<float>> A, vector<vector<float>> B) {
#//     }    
#
#       Follow up 1. For step 3. How to do the "test“? Please clarify this in your code/pseudo code
#       Follow up 2. How do decide the "k" mentioned in step 5. Think about it mathematically!
#
#       3. 【Projects】:
#       We describe this in another section.          


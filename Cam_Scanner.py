
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

im_path = 'page.jpg'
im = cv2.imread(im_path)


# In[2]:


print(im.shape)
img = cv2.resize(im, (1500,800))
print(img.shape)
plt.imshow(img)
plt.show()


# In[3]:


##image blurring
orig = img.copy()
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
plt.imshow(blurred)
plt.show()


# In[4]:


##edge detecting
edged = cv2.Canny(blurred, 0, 50)
orig_edged = edged.copy()

plt.imshow(orig_edged)
plt.show()


# In[5]:


##contours extraction
_, contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(len(contours))
contours = sorted(contours, reverse=True, key=cv2.contourArea)


# In[46]:


##best contours
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*p, True)
    
    if len(approx) == 4:
        target = approx
        break
print(target.shape)
print(target)


# In[30]:


##reorder target contour
def reorder(h):
    h = h.reshape(4,2)
    print(h)
    
    hnew = np.zeros((4,2), dtype=np.float32)
    
    add = h.sum(axis=1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    
    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    
    return hnew


# In[31]:


reordered = reorder(target)
print('---------')
print(reordered)


# In[41]:


##Project to a fixed size screen
input_represent = reordered
output_map = np.float32([[0,0], [800,0], [800,800], [0,800]])
M = cv2.getPerspectiveTransform(input_represent, output_map)


# In[42]:


ans = cv2.warpPerspective(orig, M, (800,800))


# In[43]:


plt.imshow(ans)
plt.show()


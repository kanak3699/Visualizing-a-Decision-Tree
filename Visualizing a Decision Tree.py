
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()


# In[4]:


print(iris.feature_names)


# In[5]:


print(iris.target_names)


# In[7]:


print(iris.data[0])


# In[8]:


print(iris.target[0])


# In[13]:


for i in range(len(iris.target)):
    print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))


# In[17]:


import numpy as np


# In[15]:


iris = load_iris()


# In[18]:


test_idx = [0,50,100]


# In[19]:


train_target =np.delete(iris.target, test_idx)


# In[20]:


train_data = np.delete(iris.data, test_idx, axis=0)


# In[21]:


test_target = iris.target[test_idx]


# In[23]:


test_data = iris.data[test_idx]


# In[24]:


from sklearn import tree


# In[25]:


clf = tree.DecisionTreeClassifier()


# In[26]:


clf.fit(train_data, train_target) 


# In[28]:


print(test_target)


# In[29]:


print(clf.predict(test_data))


# In[39]:


import pydotplus


# In[30]:


from sklearn.externals.six import StringIO


# In[32]:


import pydot


# In[49]:


import graphviz


# In[33]:


dot_data = StringIO()


# In[34]:


tree.export_graphviz(clf,
                        out_file=dot_data,
                        feature_names=iris.feature_names,
                        class_names=iris.target_names,
                        filled=True, rounded=True,
                        impurity=False)


# In[52]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[ ]:


graph.write_pdf("iris.pdf")


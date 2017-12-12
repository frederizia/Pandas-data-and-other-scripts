
# coding: utf-8

# # Pandas tutorial

# Following pandas tutorial from:
# https://pandas.pydata.org/pandas-docs/stable/10min.html

# In[2]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Let's create a series which is like a 1D array.

# In[3]:

s = pd.Series([1,3,5,np.nan,6,8])
s


# Pandas is smart so can deal with dates using Python's inbuilt datetime functionality. The default frequency is day, e.g. ```periods=6``` means 6 days. Let's create random data in categories A, B, C, and D corresponding to those dates.

# In[7]:

dates = pd.date_range('20130101', periods=6)
dates
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
df


# Above we created data by passing a multi-dimensional numpy array and specifying the column labels. Alternatively we can specify columns and corresponding values using a dictionary where the key is the column name and the value the data. If only one value is given it is repeated for the maximum length of the table. The values can be lists of strings, dates or numbers. A type can be specified explicitly.

# In[14]:

df2 = pd.DataFrame({ 'A' : 1.,
                    'B' : pd.Timestamp('20130102'),
                    'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                    'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo' })
df2.dtypes


# We can use pandas to give us the indices, which default to integers from 0 unless specified, column names, or just the values.

# In[17]:

df.index
df.columns
df.values


# Pandas also has a handy way of giving as an overview of the stats of a data frame.

# In[18]:

df.describe()


# Like with numpy arrays you can also transpose your data.

# In[19]:

df.T


# You can sort an array by axis or by the values in a specific column.

# In[20]:

df.sort_index(axis=1, ascending=False)


# In[23]:

df.sort_values(by='B')


# ## Indexing

# Selection values can work similarly to numpy arrays using indexing, either by indices...
# Here first three rows are selected:

# In[24]:

df[0:3]


# ...or using the index values themselves...

# In[25]:

df['20130102':'20130104']


# ... or by columns:

# In[26]:

df['A']


# More computationally efficient ways of selecting data is using pandas inbuilt functions such as ```.loc```. The first argument specifies the index by *value*, the second the columns.

# In[27]:

df.loc[dates[0]]


# In[28]:

df.loc[:,['A','B']]


# In[29]:

df.loc['20130102':'20130104',['A','B']]


# In[30]:

df.loc['20130102',['A','B']]


# If you want to get an accessible value:

# In[35]:

df.loc[dates[0],'A']


# Or equivalently

# In[36]:

df.at[dates[0],'A']


# We can also specify the index and column by *index* (integer) using ```.iloc```.

# In[37]:

df.iloc[3]


# In[40]:

df.iloc[3:5,0:2]


# In[41]:

df.iloc[[1,2,4],[0,2]]


# For selecting specific rows or columns:

# In[43]:

df.iloc[1:3,:]


# In[44]:

df.iloc[:,1:3]


# To get access to the scalar value:

# In[45]:

df.iloc[1,1]


# In[46]:

df.iat[1,1]


# ### Bolean Indexing

# Can do this by column...

# In[47]:

df[df.A > 0]


# Or by entire array which creats ```NaNs``` for values which do not conform.

# In[48]:

df[df > 0]


# We can also use ```isin()``` to pick out values out of a list.

# In[50]:

df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
df2


# The ```isin()``` method creates a boolean mask.

# In[52]:

df2['E'].isin(['two','four'])


# In[53]:

df2[df2['E'].isin(['two','four'])]


# ### Changing values in a DF

# We can change values in a dataframe by picking out cells using the methods we used above, e.g. ```.at()```,```.iat()```, ```.loc()``` etc.

# In[55]:

s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
s1


# In[58]:

df['F'] = s1
df


# Using the label...

# In[60]:

df.at[dates[0],'A'] = 0
df


# ...or position (using integer index).

# In[62]:

df.iat[0,1] = 0
df


# We can also assign an array to a column. The array needs to have the correct length.

# In[67]:

df.loc[:,'D'] = np.array([5] * len(df))
df


# We can alternatively use conditions.

# In[69]:

df2 = df.copy()
df2[df2 > 0] = -df2
df2


# ### Missing data

# When we have missing data (e.g. ```NaNs``` = ```np.nan()```) we can decide how to deal with it. This depends on what the data means, and what we're planning on doing with the data.
# Here we're adding a column to a dataframe which is by default empty.

# In[71]:

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1
df1.loc[dates[0]:dates[1],'E'] = 1
df1


# We can remove any rows that contain missing data, which here leaves only one row.

# In[72]:

df1.dropna(how='any')


# Or assign a certain value.

# In[73]:

df1.fillna(value=5)


# We can define a mask to see the missing data cells: (doesn't seem to work!)

# In[79]:

pd.isna(df1)


# ### Operations
# 
# We've learnt how to access and create data, now let's get some insight into it.
# 

# ```.mean()``` breaks down the results by column.

# In[84]:

df.mean()


# Though we can also specify a single column an access the scalar value:

# In[85]:

df['A'].mean()


# If we want to perform the operation on the other axis, i.e. the rows, we do:

# In[86]:

df.mean(1)


# Standard mathematical operations are perform
# ed through attributes, e.g.
# 
#   \+    ```.add()```
# 
#   \-    ```.sub()```
# 
#   \*    ```.mul()```
# 
#   /    ```.div()```
# 
#   %    ```.mod()```
# 
#   ^    ```.pow()```
# 
# 
# 

# In[88]:

s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
s


# In[89]:

df.sub(s, axis='index')


# ```.apply()``` can be used to apply functions to an entire column.

# In[91]:

df.apply(np.cumsum)


# We can use lambda functions, which again apply to the columns.

# In[93]:

df.apply(lambda x: x.max() - x.min())


# We can use ```.value_counts()``` to see how often each value appears. This is equivalent to histograms.

# In[94]:

s = pd.Series(np.random.randint(0, 7, size=10))
s


# In[95]:

s.value_counts()


# We can also apply standard python methods on strings.

# In[98]:

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s


# In[99]:

s.str.lower()


# ### Combining DataFrames
# 
# We can combined frames using different methods.
# 
# First let's look at ```.concat()```

# In[100]:

df = pd.DataFrame(np.random.randn(10, 4))
df


# We clan split this by row and then recombine using ```.concat()```. This simply comines the rows into one dataframe.

# In[101]:

pieces = [df[:3], df[3:7], df[7:]]
pieces


# In[102]:

pd.concat(pieces)


# Another option is merging. The behaviour depends on the structure of the data and might create additional rows.

# In[103]:

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
left


# In[104]:

right


# As here the ```key``` is the same, when we merge on key it gives us all combinations.

# In[105]:

pd.merge(left, right, on='key')


# Alternatively we have:

# In[111]:

left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
left


# In[112]:

right


# In[113]:

pd.merge(left, right, on='key')


# Appending adds to the end of a DataFrame.

# In[114]:

df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
df


# In[115]:

s = df.iloc[3]
s


# In[117]:

df.append(s, ignore_index=True)   # if we don't ignore index then there will be repeated indices


# ### Grouping
# 
# Grouping can be used for splitting data, applying operations to only some data or creating new data structures.

# In[118]:

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
df


# Group by picks out unique values in the specifiec column(s). Performing an operation is then done with respect to those unique values.

# In[119]:

df.groupby('A').sum()


# If we group by more than one column we create a hierarchical index.

# In[120]:

df.groupby(['A','B']).sum()


# ### Reshaping
# 
# We can create dataframes with several indices.

# In[123]:

tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df


# In[124]:

df2 = df[:4]
df2


# ```.stack()``` turns multiple columns into a single column with multiple rows. It essentially adds additional hierarchies.

# In[125]:

stacked = df2.stack()
stacked


# We can undo this using ```unstack()```. This automatically applies to the last level.

# In[127]:

stacked.unstack()


# In[128]:

stacked.unstack().unstack()


# We can also specify which level we want to unstack, e.g.

# In[129]:

stacked.unstack(1)


# In[130]:

stacked.unstack(0)


# There is a special feature to create **pivot tables**. We can specify *values*, *index* and *columns*. If a key-value combination does not exist we get ```NaN```.

# In[131]:

df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})
df


# In[132]:

pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


# ### Time series
# 
# Due to the inbuilt datetime functionality it is very easy to deal with time series and binning data according to different frequencies.

# In[136]:

rng = pd.date_range('1/1/2012', periods=100, freq='S')   # seconds frequency
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()


# You can also add time zones.

# In[137]:

rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts


# In[139]:

ts_utc = ts.tz_localize('UTC')
ts_utc


# In[140]:

ts_utc.tz_convert('US/Eastern')


# We can also change how the time looks.

# In[142]:

rng = pd.date_range('1/1/2012', periods=5, freq='M')  # monthly
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts


# By default the last day of the month is chosen. We can change it so that only the month is presented.

# In[143]:

ps = ts.to_period()
ps


# If we then convert this back to a time stamp we will get the first day of the month.

# In[144]:

ps.to_timestamp()


# There are a lot of things we can do with the datetime functionality.

# In[147]:

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')  # quaterly
ts = pd.Series(np.random.randn(len(prng)), prng)
ts


# In[148]:

ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts


# ### Categoricals
# 
# Categories are not numerical data and can be renamed.

# In[149]:

df = pd.DataFrame({"id":[1,2,3,4,5,6], "raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})
df


# In[150]:

df["grade"] = df["raw_grade"].astype("category")
df['grade']


# In[152]:

df["grade"].cat.categories = ["very good", "good", "very bad"]    # rename categories
df['grade']


# We can also add categories that don't appear.

# In[153]:

df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df["grade"]


# They are sorted by the order in the definition.

# In[154]:

df.sort_values(by="grade")


# In[155]:

df.groupby("grade").size()


# ## Plotting
# 
# Plotting dataframes is easy and intuitive.

# In[5]:

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts


# In[6]:

ts = ts.cumsum()
ts


# In[7]:

ts.plot()


# In[8]:

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,columns=['A', 'B', 'C', 'D'])


# In[9]:

df = df.cumsum()


# In[10]:

plt.figure(); df.plot(); plt.legend(loc='best')


# In[ ]:




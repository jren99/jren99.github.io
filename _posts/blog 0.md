Let's start from creating an empty plot.


```python
# create a single figure with a single axis (1 row x 1 column)
fig, ax = plt.subplots(1, 1)
```


    
![png](output_1_0.png)
    


Now we can add histograms of culmen length for each species on this empty plot.


```python
for s in penguins["Species"].unique():
    # select the rows with species == s
    df = penguins[penguins["Species"] == s]
    # create histogram
    plt.hist(df["Culmen Length (mm)"], label = s, alpha = 0.5)

# add legend
plt.legend()

# add x-axis
plt.xlabel("Culmen Length (mm)")

# add y-axis
plt.ylabel("Frequency")

```




    Text(0, 0.5, 'Frequency')




    
![png](output_3_1.png)
    


First, let's create a single scatterplot to observe the relationship between culmen length and culmen depth.


```python
for s in penguins["Species"].unique():
    # select the rows with species == s
    df = penguins[penguins["Species"] == s]
    # create scatter
    plt.scatter(df["Culmen Length (mm)"], df["Culmen Depth (mm)"], label = s)

# add legend
plt.legend()

# add x-axis
plt.xlabel("Culmen Length (mm)")

# add y-axis
plt.ylabel("Culmen Depth (mm)")

```




    Text(0, 0.5, 'Culmen Depth (mm)')




    
![png](output_5_1.png)
    



```python

```

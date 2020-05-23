Feature engineering is the creation of new input or target features from existing features. The objective is to create ones that do a better job of representing a machine learning problem to the model. By doing so, you can improve the accuracy of the model.
Good feature engineering can be the difference between a poor model and a fantastic one! More often than not, you will find that you can squeeze more out of your models through careful feature selection than any amount of algorithm tuning.

## Binning
Binning, (also called banding or discretisation), can be used to create new categorical features that group individuals based on the value ranges of existing features.  
You can use binning to create new target features you want to predict or new input features.
- Numerical Binning
For example, using data from the World Happiness Report, we create a new feature, happiness_band, by binning the happiness feature into low, medium, and high bands:

## code

```python
import pandas as pd
df = pd.read_csv("fe_binning.csv")
df.head()
binned = pd.cut(df['happiness'], bins=[2,4,6,10], labels=['L','M','H'])
df['happiness_band'] = binned
df.head()
df['happiness_band'].value_counts()

```
The `bins` parameter defines the boundaries of the bins. In this case, I have chosen to split the data into bins containing countries with happiness values of 2 to 4,4 to 6, and 6 to 10.

- Categorical Binning
You can also apply binning to categorical features.

## code
```python
import pandas as pd
df = pd.read_csv("fe_binning.csv")
mapping = pd.read_csv('country_region.csv')
mapping.head()
df.head()
df = pd.merge(df, mapping, on=['country', 'country'], how='left')
df.head()
df.isnull().mean()

```
# splitting

## date time decomposition
A common use of splitting is breaking dates and times into their component parts.

## code
```python
import pandas as pd
df2 = pd.read_csv('fe_splitting.csv')
df2.head()
df2.dtypes
df2['timestamp_of_call'] = pd.to_datetime(df2['timestamp_of_call'])
df2.dtypes
df2['day'] = df2['timestamp_of_call'].dt.day
df2['month'] = df2['timestamp_of_call'].dt.month
df2['year'] = df2['timestamp_of_call'].dt.year
df2['weekday'] = df2['timestamp_of_call'].dt.weekday
df2['hour'] = df2['timestamp_of_call'].dt.hour
df2.head()
df2.isnull().mean()
df2.head()

```
## Compound String Splitting
Sometimes data comes with compound strings, which are strings made up of multiple items of information. One example is in the London Fire Department data. The property_type contains information about the property type (e.g., Purpose Built Flats/Maisonette) and the size (e.g., 4 to 9 stories).

## code
```python
import pandas as pd
df2 = pd.read_csv('fe_splitting.csv')
df2.head()
df2['property_type'].unique()
df2[['property_type_type', 'property_type_size']] = df2['property_type'].str.split('-', expand=True)
df2.head()
df2.isnull().mean()

```
- One-hot encoding to deal with categorical features
- calculated feature





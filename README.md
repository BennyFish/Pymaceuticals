
## Observable Trends
1. Capomulin treatment presented a decrease in Tumor Volume by 19.4% over the 45 day span
    and is the only drug out of the 4 selected treatments that showed signs of tumor reduction.
2. The number of Metastatic sites increased in all 4 drug treatments. Even though Capomulin 
    treatment reduced tumor volume, metastatis still occured. This shows Capomulin treatment 
    is likely ineffective at stopping the spread of cancer. Capomulin treatment was able to
    target the tumor directly, but still allowed for the production of Mestastic cells.
3. The number of mice that survived the 45 day span for Infubinol was worse than the Placebo,
    while Ketapril resulted in an equal survival rate.


```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

#merge dataframes
clinical_data = pd.read_csv('raw_data/clinicaltrial_data.csv')
drug_data = pd.read_csv('raw_data/mouse_drug_data.csv')
df = pd.merge(clinical_data, drug_data, how='right', on='Mouse ID')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
      <th>Drug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b128</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b128</td>
      <td>5</td>
      <td>45.651331</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b128</td>
      <td>10</td>
      <td>43.270852</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b128</td>
      <td>15</td>
      <td>43.784893</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b128</td>
      <td>20</td>
      <td>42.731552</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
  </tbody>
</table>
</div>




```python
#set a list that contains the specific drugs we want to analyze
las_vegas = ['Capomulin','Infubinol','Ketapril','Placebo']    
```


```python
#Create a dataframe of the mean for Tumor Volume and Metastatic sites of the sample of mice for each time point
mean_df = df.groupby(['Drug','Timepoint']).mean()
mean_df = mean_df.reset_index()
mean_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Capomulin</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Capomulin</td>
      <td>5</td>
      <td>44.266086</td>
      <td>0.160000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Capomulin</td>
      <td>10</td>
      <td>43.084291</td>
      <td>0.320000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Capomulin</td>
      <td>15</td>
      <td>42.064317</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Capomulin</td>
      <td>20</td>
      <td>40.716325</td>
      <td>0.652174</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Create a dataframe of the Standard Error of the Mean for Tumor Volume and 
#   Metastatic sits for the sample of mice for each Timepoint. This provides us with our error bars.
sem_df = df.groupby(['Drug','Timepoint']).sem()
sem_df = sem_df.drop(columns='Mouse ID', axis=1).reset_index()
sem_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Metastatic Sites</th>
      <th>Tumor Volume (mm3)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Capomulin</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Capomulin</td>
      <td>5</td>
      <td>0.074833</td>
      <td>0.448593</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Capomulin</td>
      <td>10</td>
      <td>0.125433</td>
      <td>0.702684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Capomulin</td>
      <td>15</td>
      <td>0.132048</td>
      <td>0.838617</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Capomulin</td>
      <td>20</td>
      <td>0.161621</td>
      <td>0.909731</td>
    </tr>
  </tbody>
</table>
</div>




```python
mean_pivot1 = mean_df.pivot(index='Timepoint', columns='Drug',values='Tumor Volume (mm3)')
mean_pivot = mean_pivot1.reset_index()
sem_pivot = sem_df.pivot(index='Timepoint', columns='Drug',values='Tumor Volume (mm3)')
sem_pivot = sem_pivot.reset_index()
```


```python
plt.figure(figsize=(8,5))

plt.xlim(0,45)
plt.ylim(30,75)
plt.title("Tumor Response to Treatment", fontsize=15)
plt.xlabel('Time (Days)', fontsize=15)
plt.ylabel('Tumor Volume (mm3)', fontsize=15)
plt.grid(color='k', linestyle=':', linewidth=.5)

#Loop through list of drugs to plot
for fear_and_loathing in las_vegas:
    plt.errorbar(mean_pivot.Timepoint, mean_pivot[fear_and_loathing], 
                 sem_pivot[fear_and_loathing], fmt='2', 
                 linestyle='--', marker='.', markersize=8,capsize=4)                
plt.legend(mean_pivot[las_vegas])
plt.show()
```


![png](output_6_0.png)



```python
#Using same grouby as above, I adjusted the code to look towards the Metastatic Sites column.
meta_mean_pivot = mean_df.pivot(index='Timepoint', columns='Drug',values='Metastatic Sites')
meta_mean_pivot = meta_mean_pivot.reset_index()
meta_sem_pivot = sem_df.pivot(index='Timepoint', columns='Drug',values='Metastatic Sites')
meta_sem_pivot = meta_sem_pivot.reset_index()
```


```python
plt.figure(figsize=(8,5))
plt.xlim(0,45)
plt.ylim(0,4)
plt.title("Metastatic Sites vs Time", fontsize=15)
plt.xlabel('Time (Days)', fontsize=15)
plt.ylabel('Metastatic Sites', fontsize=15)
plt.grid(color='k', linestyle=':', linewidth=.5)

#Loop through list of drugs to plot
for fear_and_loathing in las_vegas:
    plt.errorbar(meta_mean_pivot.Timepoint, meta_mean_pivot[fear_and_loathing], 
                 meta_sem_pivot[fear_and_loathing], fmt='2', linestyle='--', 
                 marker='.', markersize=8, capsize=4)              
plt.legend(meta_mean_pivot[las_vegas])
plt.show()
```


![png](output_8_0.png)



```python
#Create a Dataframe of the average number of mice for each drug at each timepoint.
mouse_count_df = df.groupby(['Drug','Timepoint']).count()
mouse_count_df = mouse_count_df.drop(columns={'Tumor Volume (mm3)',
                                              'Metastatic Sites'}).reset_index()
mouse_count_pivot = mouse_count_df.pivot(index='Timepoint', columns='Drug', 
                                        values='Mouse ID')
mouse_count_pivot = (mouse_count_pivot/mouse_count_pivot.iloc[0]*100).reset_index()
mouse_count_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.000000</td>
      <td>100.0</td>
      <td>100.000000</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>100.0</td>
      <td>84.0</td>
      <td>100.0</td>
      <td>92.0</td>
      <td>92.0</td>
      <td>96.0</td>
      <td>96.153846</td>
      <td>100.0</td>
      <td>96.153846</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>100.0</td>
      <td>80.0</td>
      <td>84.0</td>
      <td>88.0</td>
      <td>84.0</td>
      <td>96.0</td>
      <td>88.461538</td>
      <td>96.0</td>
      <td>88.461538</td>
      <td>88.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>96.0</td>
      <td>76.0</td>
      <td>84.0</td>
      <td>76.0</td>
      <td>84.0</td>
      <td>80.0</td>
      <td>65.384615</td>
      <td>96.0</td>
      <td>88.461538</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>92.0</td>
      <td>72.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>80.0</td>
      <td>76.0</td>
      <td>65.384615</td>
      <td>92.0</td>
      <td>80.769231</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>88.0</td>
      <td>72.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>72.0</td>
      <td>68.0</td>
      <td>53.846154</td>
      <td>92.0</td>
      <td>73.076923</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30</td>
      <td>88.0</td>
      <td>64.0</td>
      <td>68.0</td>
      <td>72.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>50.000000</td>
      <td>92.0</td>
      <td>69.230769</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35</td>
      <td>88.0</td>
      <td>56.0</td>
      <td>48.0</td>
      <td>68.0</td>
      <td>60.0</td>
      <td>56.0</td>
      <td>38.461538</td>
      <td>84.0</td>
      <td>61.538462</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>40</td>
      <td>84.0</td>
      <td>56.0</td>
      <td>40.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>48.0</td>
      <td>34.615385</td>
      <td>80.0</td>
      <td>46.153846</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>45</td>
      <td>84.0</td>
      <td>52.0</td>
      <td>36.0</td>
      <td>44.0</td>
      <td>52.0</td>
      <td>44.0</td>
      <td>26.923077</td>
      <td>80.0</td>
      <td>42.307692</td>
      <td>56.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#plot the dataframe for the selected treatments
plt.figure(figsize=(8,5))

plt.xlim(0,df.Timepoint.max())
plt.ylim(30,105)
plt.title("Survival Rate per Treatment", fontsize=15)
plt.xlabel('Time (Days)', fontsize=15)
plt.ylabel('Survival Rate (%)', fontsize=15)
plt.grid(color='k', linestyle=':', linewidth=.5)

#Loop through list of drugs to plot
for fear_and_loathing in las_vegas:
    plt.errorbar(mouse_count_pivot.Timepoint, mouse_count_pivot[fear_and_loathing],
                linestyle='--', marker=9)
                
plt.legend(mouse_count_pivot[las_vegas])
plt.show()
```


![png](output_10_0.png)



```python
change_mean_pivot = mean_pivot1 - mean_pivot1.shift(1)
change_series = change_mean_pivot.sum()/mean_pivot1.iloc[0]*100
```


```python
height=[]
for i, v in change_series.iteritems():
    for fear_and_loathing in las_vegas:
        if fear_and_loathing == i:
            height.append(v)
```


```python
plt.ylim(-20,60)
plt.title("Tumor Volume Over 45 Day Treatment", fontsize=10)
plt.ylabel('% Tumor Volume Change', fontsize=10)
plt.grid(color='k', linestyle=':', linewidth=.5)
#clrs  = [clrred if pwval[x] >= pwlim[x] else clrgrn for x in range(ndays)]
red = 'red'
green = 'green'
colors = [red if height[x] > 0 else green for x in range(0,len(height))]
tumor_bar = plt.bar(las_vegas, height, align='center', alpha=0.5, color=colors)

def sizelabel(bars):
     for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x()+ bar.get_width()/2.,height-height,'%d' % int(height) +"%",
                ha='center', va='baseline')
sizelabel(tumor_bar)
```


![png](output_13_0.png)


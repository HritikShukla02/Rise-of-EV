# Rise-of-EV

## **Introduction**  

The global automotive landscape is undergoing a significant shift, with increasing consumer interest in **electric vehicles (EVs)** driven by advancements in technology, environmental concerns, and government incentives. In India, this transformation has been particularly notable in recent years, as more consumers are transitioning from **traditional fuel-powered vehicles (petrol and diesel) to EVs and hybrid alternatives**.  

This analysis aims to explore the **growing adoption of EVs** by examining vehicle registration trends across different fuel types from **2016 to 2024**. By leveraging official registration data from **vahan.parivahan.gov.in**, this study seeks to highlight the shift in consumer preferences and quantify the increasing market penetration of EVs.  

### **Objective**  
The primary focus of this study is to **demonstrate the rising interest in EVs** and analyze:  
- The **year-over-year growth** of EV registrations compared to conventional fuel types.  
- **Key milestones** that influenced EV adoption (e.g., government policies, major EV launches).  
- The **rate of decline** (if any) in petrol/diesel vehicle registrations.  

### **Methodology**  
This analysis will use **descriptive statistics, trend visualization, and comparative analysis** to identify patterns in fuel-type preferences. The data-driven insights will support discussions on the **market shift towards electric mobility** and provide empirical evidence for the increasing consumer demand for EVs.  

### **Sections in this Analysis**
- Preprocessing
- Understanding the dataset
- Analysing the Key Trends
- Findings

<hr>

## **Understanding the Dataset**  

This analysis is based on data sourced from **vahan.parivahan.gov.in**, which serves as India's official vehicle registration database. The dataset provides a comprehensive record of registered vehicles across the country, along with details such as **total transactions and revenue generated** over the years.  

### **Dataset Overview**  
The dataset focuses on the **total number of vehicles registered in India from 2016 to 2024**, categorized by their **fuel type**. It enables a year-over-year comparison of vehicle registrations to track consumer preferences and market trends.  

### **Key Features of the Dataset**  
- **Time Period:** 2016 to 2024  
- **Categories:** Vehicle registrations based on **fuel type**  
- **Fuel Types Included:**  
  - **Electric Vehicles (EVs):** PURE EV, PLUG-IN HYBRID EV, STRONG HYBRID EV, ELECTRIC (BOV)  
  - **Conventional Fuel Vehicles:** Petrol, Diesel, CNG, LPG, and their hybrid variants  
- **Data Structure:**  
  - **Columns:** Represent years (2016–2024)  
  - **Rows:** Represent fuel types  
  - **Values:** Number of vehicles registered per year for each fuel type  

### **Key Notes**
- **Early Years Show Minimal Registrations:** Vehicle registration numbers were extremely low (mostly 0) in the earlier years.  
- **Rise in ELECTRIC(BOV) Registrations:** A noticeable increase in **ELECTRIC(BOV) registrations** in later years could be attributed to the growing adoption of **e-Rickshaws**.  
- **PURE EV Growth in 2024:** The surge in **PURE EV registrations in 2024** aligns with the launch of **TATA and Mahindra’s first PURE EVs in January and November 2024**, marking a significant shift towards electric mobility.  
- **Exclusion of 2025 Data:** Data for the year **2025 was not included in the analysis** as it might present an unclear picture due to incomplete or evolving registration trends.

### **Significance of the Dataset**  
This dataset allows us to:  
1. **Analyze the shift in consumer preference** from conventional fuel vehicles to EVs.  
2. **Identify trends in EV adoption** and compare their growth against traditional fuel types.   

This dataset provides a solid foundation for understanding India's evolving **automobile market dynamics**, particularly the increasing interest in **electric mobility**.  

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fuel Type</th>
      <th>2024</th>
      <th>2023</th>
      <th>2022</th>
      <th>2021</th>
      <th>2020</th>
      <th>2019</th>
      <th>2018</th>
      <th>2017</th>
      <th>2016</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>CNG ONLY</td>
      <td>482363</td>
      <td>421341</td>
      <td>297450</td>
      <td>166339</td>
      <td>43257</td>
      <td>33539</td>
      <td>31498</td>
      <td>26269</td>
      <td>30256</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DIESEL</td>
      <td>2638300</td>
      <td>2569397</td>
      <td>2346314</td>
      <td>2014421</td>
      <td>2025883</td>
      <td>2926791</td>
      <td>3183671</td>
      <td>2867205</td>
      <td>2752005</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DIESEL/HYBRID</td>
      <td>5249</td>
      <td>2462</td>
      <td>737</td>
      <td>35</td>
      <td>2917</td>
      <td>39108</td>
      <td>59554</td>
      <td>48500</td>
      <td>11351</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DUAL DIESEL/BIO CNG</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>DUAL DIESEL/CNG</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>15</td>
      <td>1</td>
      <td>7</td>
      <td>5</td>
      <td>4</td>
      <td>14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DUAL DIESEL/LNG</td>
      <td>21</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>166885</td>
      <td>130243</td>
      <td>87389</td>
      <td>49825</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ELECTRIC(BOV)</td>
      <td>1457348</td>
      <td>1532423</td>
      <td>1024996</td>
      <td>332371</td>
      <td>124672</td>
      <td>8</td>
      <td>17</td>
      <td>22</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ETHANOL</td>
      <td>183</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FUEL CELL HYDROGEN</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LNG</td>
      <td>349</td>
      <td>244</td>
      <td>20</td>
      <td>3</td>
      <td>14</td>
      <td>3866</td>
      <td>4067</td>
      <td>3813</td>
      <td>3969</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LPG ONLY</td>
      <td>32637</td>
      <td>26812</td>
      <td>14404</td>
      <td>13410</td>
      <td>9912</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>METHANOL</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>89243</td>
      <td>114268</td>
      <td>108575</td>
      <td>130409</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NOT APPLICABLE</td>
      <td>63160</td>
      <td>66379</td>
      <td>80469</td>
      <td>82529</td>
      <td>84081</td>
      <td>20291107</td>
      <td>21297715</td>
      <td>19725544</td>
      <td>17924529</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PETROL</td>
      <td>19290384</td>
      <td>18131493</td>
      <td>17178363</td>
      <td>15880107</td>
      <td>15975775</td>
      <td>421331</td>
      <td>463465</td>
      <td>346070</td>
      <td>337314</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PETROL/CNG</td>
      <td>729461</td>
      <td>543579</td>
      <td>433509</td>
      <td>279199</td>
      <td>240189</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PETROL/ETHANOL</td>
      <td>656923</td>
      <td>379863</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>76169</td>
      <td>11749</td>
      <td>709</td>
      <td>530</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PETROL/HYBRID</td>
      <td>295083</td>
      <td>331239</td>
      <td>192295</td>
      <td>125648</td>
      <td>86218</td>
      <td>109314</td>
      <td>124160</td>
      <td>88195</td>
      <td>82645</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PETROL/LPG</td>
      <td>4976</td>
      <td>5881</td>
      <td>12109</td>
      <td>20382</td>
      <td>40275</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PLUG-IN HYBRID EV</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>64</td>
      <td>139</td>
      <td>234</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PURE EV</td>
      <td>493178</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>SOLAR</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>STRONG HYBRID EV</td>
      <td>58263</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<hr>

### **Data Preprocessing**  
Before conducting the analysis, the dataset was preprocessed to:  
- Convert numerical values into a consistent format (e.g., removing commas from numbers).  
- Handle missing or incomplete data. 
- Deciding broader themes based on Fuels.
    - Theme 1: (EV) 
        - PURE EV , ELECTRIC(BOV), PLUG-IN HYBRID EV, STRONG HYBRID EV
    - Theme 2: (Hybrid)
        - DIESEL/HYBRID, PETROL/HYBRID
    - Theme 3: (GAS)
        - CNG ONLY, LPG ONLY, PETROL/CNG, PETROL/LPG, DUAL DIESEL/CNG
    - Theme 4: Traditional Fuels (Studied Separately)
        - Petrol, Diesels
    - Other Fuels
- Transform the wide-format table into a long-format structure for effective visualization. (This step was taken later in the analysis)

Final Data after Preprocessing:
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>EV</th>
      <th>Hybrid</th>
      <th>Gas</th>
      <th>Petrol</th>
      <th>Diesel</th>
      <th>Others</th>
      <th>Total</th>
      <th>EV_pct</th>
      <th>Hybrid_pct</th>
      <th>Gas_pct</th>
      <th>Petrol_pct</th>
      <th>Diesel_pct</th>
      <th>Others_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024</td>
      <td>2008831</td>
      <td>300332</td>
      <td>1249437</td>
      <td>19290384</td>
      <td>2638300</td>
      <td>720652</td>
      <td>26207936</td>
      <td>7.664972</td>
      <td>1.145958</td>
      <td>4.767399</td>
      <td>73.605125</td>
      <td>10.066798</td>
      <td>2.749747</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023</td>
      <td>1532433</td>
      <td>333701</td>
      <td>997613</td>
      <td>18131493</td>
      <td>2569397</td>
      <td>446486</td>
      <td>24011123</td>
      <td>6.38218</td>
      <td>1.389777</td>
      <td>4.154795</td>
      <td>75.51289</td>
      <td>10.700861</td>
      <td>1.859497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022</td>
      <td>1024997</td>
      <td>193032</td>
      <td>757477</td>
      <td>17178363</td>
      <td>2346314</td>
      <td>80493</td>
      <td>21580676</td>
      <td>4.749606</td>
      <td>0.894467</td>
      <td>3.509978</td>
      <td>79.600671</td>
      <td>10.872291</td>
      <td>0.372986</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021</td>
      <td>332371</td>
      <td>125683</td>
      <td>479345</td>
      <td>15880107</td>
      <td>2014421</td>
      <td>82536</td>
      <td>18914463</td>
      <td>1.757232</td>
      <td>0.664481</td>
      <td>2.534278</td>
      <td>83.957483</td>
      <td>10.650162</td>
      <td>0.436364</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020</td>
      <td>124672</td>
      <td>89135</td>
      <td>333634</td>
      <td>15975775</td>
      <td>2025883</td>
      <td>84105</td>
      <td>18633204</td>
      <td>0.669085</td>
      <td>0.478366</td>
      <td>1.790535</td>
      <td>85.738207</td>
      <td>10.872435</td>
      <td>0.451372</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019</td>
      <td>9</td>
      <td>148422</td>
      <td>33552</td>
      <td>421331</td>
      <td>2926791</td>
      <td>20627277</td>
      <td>24157382</td>
      <td>0.000037</td>
      <td>0.614396</td>
      <td>0.138889</td>
      <td>1.744109</td>
      <td>12.115514</td>
      <td>85.387055</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018</td>
      <td>82</td>
      <td>183714</td>
      <td>31504</td>
      <td>463465</td>
      <td>3183671</td>
      <td>21558042</td>
      <td>25420478</td>
      <td>0.000323</td>
      <td>0.722701</td>
      <td>0.123932</td>
      <td>1.823195</td>
      <td>12.524041</td>
      <td>84.805809</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2017</td>
      <td>161</td>
      <td>136695</td>
      <td>26278</td>
      <td>346070</td>
      <td>2867205</td>
      <td>19926033</td>
      <td>23302442</td>
      <td>0.000691</td>
      <td>0.586612</td>
      <td>0.112769</td>
      <td>1.485123</td>
      <td>12.304311</td>
      <td>85.510493</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016</td>
      <td>237</td>
      <td>93996</td>
      <td>30274</td>
      <td>337314</td>
      <td>2752005</td>
      <td>18109268</td>
      <td>21323094</td>
      <td>0.001111</td>
      <td>0.440818</td>
      <td>0.141978</td>
      <td>1.581919</td>
      <td>12.906218</td>
      <td>84.927957</td>
    </tr>
  </tbody>
</table>
</div>

<hr>

## Analysing the key trends

- ##### EV Growth Over Time:

    - <p> Bar chart showing the dramatic increase in electric vehicle registrations from 2016 to 2024
    - <p> Includes data labels for each year
![alt text](./data/plots/ev_growth.png)
<br>
The graph above shows a massive surge in EV registrations in India from 2020 onward, with numbers skyrocketing from 124K in 2020 to over 2 million in 2024. EV adoption was minimal before 2019 but gained momentum due to government incentives, lower battery costs, and rising fuel prices. The steep growth post-2021 highlights a shift in consumer preference towards sustainable mobility. Policy support, infrastructure development, and popular EV models like Tata Nexon EV have fueled this boom. This trend signals India’s rapid transition to electric vehicles for a greener future. 🚀
<br>
<br>

- ##### EV Adoption Rate:

    - <p> Line chart showing the percentage of EVs among all vehicle registrations
    - <p> Helps identify the acceleration points in market penetration
![alt text](./data/plots/ev_adoption_rate.png)
<br>
This graph illustrates India's EV adoption rate as a percentage of total vehicle registrations from 2016 to 2024. The adoption remained negligible until 2019 but started rising in 2020, reaching 7.66% by 2024. The sharp increase from 0.67% in 2020 to 4.75% in 2022 highlights a major shift towards EVs. Government policies, subsidies, rising fuel costs, and better EV infrastructure have significantly influenced this growth. The trend suggests that EVs are rapidly gaining mainstream acceptance in India. 🚗⚡
<br>
<br>

- ##### Fuel Type Comparison (2016 vs 2024):

    - <p> Side-by-side bar chart comparing the number of registrations by fuel type between 2016 and 2024
    - <p> Highlights the dramatic shift in the vehicle landscape
![alt text](./data/plots/fuel_type_comparison.png)
<br>
This bar chart compares vehicle types in 2016 vs. 2024 in India. EVs have surged significantly, while petrol vehicles remain dominant but have slightly declined. Diesel vehicle numbers are mostly stable, while hybrid and gas-powered vehicles have seen modest growth. The trend suggests a shift towards electrification and alternative fuels, reflecting government incentives, policy shifts, and consumer demand for sustainable transport. 🚗⚡
<br>
<br>

- ##### Market Share (2024):

    - <p> Pie chart comparing the proportion of market shares of vehicles by fuel types in 2024
    - <p> Highlights the current leader of fuel types in the vehicle landscape
![alt text](./data/plots/market_share_2024.png)
<br>
<p> This pie chart illustrates the vehicle registration market share for 2024. Petrol vehicles still dominate with 73.6% of the market, but EVs have grown to 7.7%, showing an increasing shift towards electrification. Diesel vehicles make up 10.1%, while gas-powered and hybrid vehicles have smaller shares at 4.8% and 1.1%, respectively. The "Others" category holds 2.7%.
<p> This highlights a gradual transition towards alternative fuels, with EV adoption rising, though petrol remains the most common fuel type. 🚗⚡
<br>
<br>

- ##### Evolution of EV Types:
    - <p> Stacked bar chart showing the breakdown of different EV technologies over time
    - <p> Shows how the EV market has diversified from ELECTRIC(BOV) to include PURE EV, PLUG-IN HYBRID, etc.
![alt text](./data/plots/evolution.png)
<br>
This **grouped bar chart** illustrates the **vehicle registrations by year and fuel type**, showing a significant rise in **electric (BOV) vehicle registrations** from 2020 to 2024.  

Key takeaways:  
- **Electric (BOV) registrations** have surged, reaching over **1.5 million in 2023** before slightly dropping in 2024.  
- **PURE EV registrations** started appearing in 2024, reaching around **500,000 units**.  
- **Plug-in hybrid EVs and strong hybrid EVs** have minimal presence, indicating that the market is primarily shifting towards fully electric vehicles rather than hybrids.  

This trend suggests a growing **adoption of electric vehicles (EVs)**, with **2023 being a peak year** for new EV registrations. 🚗⚡
<br>
<hr>

### **Key Findings**  

#### **1. Petrol Dominates the Market (Pie Chart)**
- **Petrol vehicles hold a massive 73.6% share** in 2024, making them the most registered vehicle type.  
- Diesel vehicles come in second with **10.1%**, showing a declining trend in fossil-fuel reliance.  
- **EVs (7.7%)** are growing but still significantly lower compared to petrol vehicles.  
- **Hybrid and gas vehicles** have minimal adoption, with just **1.1% and 4.8%**, respectively.  
- **"Others" category (2.7%)** suggests emerging alternative fuel types but not yet significant.

#### **2. Rapid Growth in EV Registrations (Bar Chart)**
-  **Early Years Show Minimal Registrations:** Vehicle registration numbers remained **extremely low (mostly 0s) in earlier years**, indicating a lack of adoption may be because of poor options or simply because of data availability. 
- **Electric (BOV) vehicles** saw a massive surge in registrations from **2020 to 2023**, peaking at **over 1.5 million in 2023**. This was likely driven by the **increasing adoption of e-Rickshaws**.  
- There was a slight dip in **2024**, possibly due to **market saturation or policy changes**.  
- **PURE EVs entered the market in 2024** with **over 500,000 registrations**, showing a rise in consumer preference for full electric models.  The reason behind this could be the **launch of TATA and Mahindra’s first PURE EVs in January and November 2024**, highlighting a major shift towards electric mobility. 
- As of 2024 Dec **Plug-in hybrid EVs and strong hybrid EVs** remain **very low in adoption**, indicating the shift towards fully electric models rather than hybrids.

#### **3. Shift Away from Diesel and Gas Vehicles**
- Diesel and gas vehicle shares are **low and declining**, suggesting a market shift towards **cleaner energy sources**.  
- The increasing **EV adoption aligns with global trends** toward sustainability and government incentives for clean energy vehicles. 
<hr> 

### **Conclusion**  

The analysis reflects a **gradual but significant transition towards electric vehicles**, especially from **2020 onwards**. While **petrol vehicles continue to dominate**, their share is expected to **decline over time** as EV adoption increases.  

Key trends observed:  
- **ELECTRIC(BOV) registrations** have grown steadily, likely due to the **rise in e-Rickshaws**.  
- **PURE EV registrations surged in 2024**, aligning with the launch of **TATA and Mahindra’s first PURE EVs** in January and November 2024.  
- **Diesel and gas vehicles are declining**, making way for **EVs and alternative fuel sources**.  
- **2023 marked the peak year for EV growth so far**, with increasing adoption expected in the coming years.  

Additionally, **2025 data was intentionally excluded** to prevent potential misinterpretation due to **incomplete or evolving registration data**.  

📊 **Overall, the market is shifting towards electrification, but petrol vehicles still hold a strong position in 2024. Future growth will depend on policy incentives, charging infrastructure, and battery advancements.** 🚗⚡
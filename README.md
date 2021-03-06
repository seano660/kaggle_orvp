## <b>kaggle_orvp</b>

<i>Author: Sean O'Hara</i>

A simulation-based approach to predicting stock volatility by modeling price movements as a system of competing queues. <i><u>Still a work in progress!</u></i>

<b>** Note:</b> Training and running the model requires data from Kaggle's Optiver Realized Volatility Prediction competition. The data can be downloaded [here](https://www.kaggle.com/c/optiver-realized-volatility-prediction/data). 

Tags: Simulation, Optimization, Data Engineering, Model Design

### Table of Contents
I. [Background](#sec_1)

II. [Literature](#sec_2)

III. [Data Engineering](#sec_3)

IV. [Model Construction](#sec_4)

V. [Performance Analysis](#sec_5)

VI. [Conclusion](#sec_6)

<a id = sec_1></a>
## I. Background

### Overview 
The motivation of this project stems from [Optiver's Realized Volatility Prediction challenge](https://www.kaggle.com/c/optiver-realized-volatility-prediction/overview), a Kaggle Coding Competition launched in June 2021. 

The competition challenge is to predict a stock's realized volatility over a 10-minute span given data from the preceding ten minutes -- namely second-by-second snapshots of the order book as well as any trades occurring over the period. 

The sections below give a quick overview of key calculations and data used in the project; more detailed information can be found in the competition's [tutorial notebook](https://www.kaggle.com/jiashenliu/introduction-to-financial-concepts-and-data?scriptVersionId=67183666#Competition-data) and the [data page](https://www.kaggle.com/c/optiver-realized-volatility-prediction/data). 

### Key Calculations

A stock's **realized volatility**, $\sigma$, is determined by the following expression:

$$
\sigma = \sqrt{\sum_{t}r_{t_2, t_1}^2}
$$

, where $r_{t_2, t_1}$ represents the log return between $t_{1}$ and $t_{2}$. 

**Log return** $r$ is simply the logarithm of the ratio between the ending and starting prices over some interval:

$$
r_{t_2, t_1} = \log \left( \frac{S_{t_2}}{S_{t_1}} \right)
$$

, $S_{t}$ being the weighted average price of the stock at time $t$; $0 \le t_{1} \lt t_{2} \lt 600$.

**Weighted average price (WAP)** is the calculation used to determine a stock's price from the most competitive bid and ask quotes. 

$$ WAP = \frac{BidPrice_{1}*AskSize_{1} + AskPrice_{1}*BidSize_{1}}{BidSize_{1} + AskSize_{1}} $$


### Data

There are two (linked) datasets given in the competition: book data and trade data. 

<b>Book data</b> consists of second-by-second snapshots of the order book at the two most competitive price levels (for both bid and ask). A sample row of data can be found below: 

| stock_id | time_id | seconds_in_bucket | bid_price1 | ask_price1 | bid_price2 | ask_price2 | bid_size1 | ask_size1 | bid_size2 | ask_size2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 0 | 1.001422 | 1.002301 | 1.00137 | 1.002353 | 3 | 226 | 2 | 100 |

The dataset contains the first and second-most competitive bid and ask quotes (suffixed 1 and 2, respectively). Prices are scaled to (approximately) unit magnitude at the outset of the 10-minute window. 

<b>Trade data</b> similarly consists of second-by-second snapshots of any actual trades occurring in the period. Below is a sample row of data: 

| stock_id | time_id | seconds_in_bucket | price | size | order_count |
| --- | --- | --- | --- | --- | --- |
| 0 | 5 | 21 | 1.002301 | 326 | 12 |

Any trades that occur are aggregated over a 1-second interval. In the above example, 12 separate trades occurred between t = 4 and t = 5, resulting in a total of 326 shares exchanged at a weighted average price of 1.002301. 

For both data sets, `time_id` corresponds to a 10-minute period of data for the given `stock_id`. `seconds_in_bucket` indicates the time elapsed in the 10-minute window. There are generally much fewer than 600 rows per `time_id`-`seconds_in_bucket` combination, as data is only reported if there is some activity since the last tick. 

In the training data, there are **112** unique stock_ids with roughly **3,830** time_ids each. 

<a id = "sec_2"></a>
## II. Approach & Relevant Literature

### Overview 

The approach to modeling volatility in this project is actually to not model it -- at least not directly. 

Instead, the objective is to simulate the price time series, and then convert to volatility using the calculations discussed in Section I. 

This fundamental design choice is driven by three observations:
1. <b>The given data is more suited to modeling price.</b> There is no obvious, natural linkage between volatility and the first two levels of the order book. Prices, on the other hand, are determined directly from the order book data. 
2. <b>Realized volatility is a first-order effect of price.</b> In other words, volatility is directly determined by observed prices. Since this is the case, if it is possible to reconstruct price from the given data, it makes little sense to model volatility as some function of other predictors.
3. <b>Realized volatility is by definition nonlinear.</b> In the case of a pure volatility model, this would require some embedding of the functional form into the model itself. Modeling price requires no such adjustment. 

It should be noted that this is somewhat of a unique approach in the competition field: most public solutions attempt to predict volatility <i>directly</i> from the given data and rely largely on blackbox methods such as neural networks and LightGBM.

A survey of contemporary  methods in <i>Statistical Modeling of High Frequency Financial Data</i> (Cont 2011) shows that a popular approach to modeling prices is to treat the underlying order book as a system of queues, with one queue to represent bid orders and another to represent ask orders (19). By modeling the six event types that act upon the order book -- market buy/sell, limit buy/sell, and cancel buy/sell -- it is possible to produce a model that captures short-term price dynamics. 

The approach in this project is based on the limit order book modeling described in Cont 2011, among others. The overarching idea is to model the six event types from historical data, and then use them to simulate the limit order book in Monte Carlo fashion -- generating hundreds of possible outcomes and then aggregating them to produce a central/most likely prediction.

### Detailed Approach

<u>Version 1 of the Limit Order Book model (LOB1.0)</u> is a simplified, proof-of-concept model that aims to show the viability of the simulation framework in predicting realized volatility. It is defined by the following specifications: 

1. <b>A focus on only the most competitive bid and ask levels.</b> By limiting the space at which events can occur, this simplification drastically reduces the complexity of the order book model and should improve both interpretability and runtime. Empirical studies also find that the majority of order flow (and by extension price movement) occurs at the most competitive bid and ask levels - so it is likely that performance is not materially harmed. 

2. <b>The assumption that events are independent Poisson processes.</b> Exploratory analysis of the event data shows that the interarrival times by event tend to be exponentially distributed. By the assumption of independence, the event generation process is also greatly simplified. But this design choice is also one that in all likelihood harms performance. Numerous studies show that events tend to be both self-exciting (an event increases the arrival rate of subsequent events of the same type) and mutually exciting (an event can drive other events). 

3. <b>A constant order size for each event type.</b> It should be noted that order size generally fluctuates quite substantially and is often dependent on the shares available at a particular price level.

4. <b>A constant shift in the order book when a price level is exhausted.</b> This means that when all shares at the most competitive bid (ask) level are depleted, the <i>entire</i> order book will shift in the downward (upward) direction. This is done in the interest of maintaining a constant bid-ask spread, and also captures the strong empirical correlation between bid and ask prices. 

4. <b>A separate model for each stock.</b> It is hypothesized that the stocks in the training sample differ materially in terms of their average trading volume and general price behavior. A global model trained on all stocks would likely fail to capture these differences. Since roughly 3,830 training instances exist for each individual stock, it is reasoned that there is enough data to model each stock individually, which should more effectively capture individual stock dynamics. 

5. <b>A global-local model.</b> It is assumed that there is substantial variance of event parameters (i.e. frequency, size) over each time_id snapshot, even if restricting the model to a single stock. For this reason, model parameters are calculated as a mixture of the local (i.e. time_id-specific) and global, macro-aggregated values. The mixture coefficient $\alpha$ is determined through Bayesian optimization.

<a id = "sec_3"></a>
## III. Data Engineering

The limit order book model is defined by the six event types and how they interact. As such, the first major step in building the model is to deconstruct the given order book and trade data into the underlying events that took place. 

The raw trade data is already well suited for this task, given that each row corresponds to a market buy or sell event. In contrast, for each row of book data there can be anywhere from zero to four events (an event is possible at each price level), necessitating a more complex algorithm. Therefore, the this section will focus on how the book data is processed to yield a comprehensive event record. 

<b>Note:</b> To avoid loading the entire dataset (>2 GB in size) into memory, all processing is conducted on a stock-by-stock basis. 

### Preprocessing

Before any intensive data manipulation can be done, the raw input data must be cleaned. There are several steps that take place in this respect, but the most critical is the reindexing of the book data. As mentioned briefly in Section I, the raw book data only contains entries where there is change to at least one of the price levels. Of course, the timing of these changes will certainly differ for each 10-minute window of data, and therefore in the interest of ensuring a stable and performant algorithm it is necessary to reindex the data.

```python 
data = data.reindex(
    labels = pd.MultiIndex.from_product(
        [data.index.get_level_values(0).unique(), range(0, 600)], 
        names = ["time_id", "seconds_in_bucket"]), 
    method = "ffill"
)
```

This code snippet is responsible for the reindexing operation, ensuring that every `time_id` has all six hundred possible `seconds_in_bucket` entries. Any missing rows are also forward filled with data from the most recent book update (they would be NA otherwise!) . 

### Re-engineering 

The next piece of the data engineering pipeline is to refactor and filter the order book data into a list of potential events that can be easily interpreted by the event tagging algorithm.

It is possible to detect that an event has occurred by looking at changes in the order size at each outstanding price level. To this end, the data is first transformed from the initial wide-form representation (where each row represents four price levels) to a long-form representation where a single price level is represented at each row. 

Before: 

| stock_id | time_id | seconds_in_bucket | bid_price1 | ask_price1 | bid_price2 | ask_price2 | bid_size1 | ask_size1 | bid_size2 | ask_size2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 0 | 1.001422 | 1.002301 | 1.00137 | 1.002353 | 3 | 226 | 2 | 100 |

After:

| stock_id | time_id | seconds_in_bucket | order_type | price | size |
| --- | --- | --- | --- | --- | --- |
| 0 | 5 | 0 | bid | 1.001422 | 3 |
| 0 | 5 | 0 | bid | 1.00137 | 2 |
| 0 | 5 | 0 | ask | 1.002301 | 226 |
| 0 | 5 | 0 | ask | 1.002353 | 100 |

The long-form representation makes it much easier to identify the outstanding orders at each price level; by extension, tracking order size changes also becomes more straightforward. This is exactly what is done next: for each row in the data, simply determine the change in size from the previous second, filling in zeros for missing entries. The result of this operation, after filtering out all changes of size 0, is something like the following: 

| stock_id | time_id | seconds_in_bucket | order_type | price | size_change |
| --- | --- | --- | --- | --- | --- |
| 0 | 5 | 1 | ask | 1.002301 | -126 |
| 0 | 5 | 5 | ask | 1.002353 | -100 |
| 0 | 5 | 6 | ask | 1.002301 | 26 |


### Event tagging

The event tagging algorithm labels each row of the engineered book and trade data with the correct order event type. The method for determining the correct order type is briefly discussed below. 

#### From book data: 
<ul>An event is classified as a <b>limit buy</b> if it occurred at a bid price level with size change > 0.</ul>
<ul>An event is classified as a <b>cancel buy</b> if it occurred at a bid price level with size change < 0.</ul>
<ul>An event is classified as a <b>limit sell</b> if it occurred at an ask price level with size change > 0.</ul>
<ul>An event is classified as a <b>cancel sell</b> if it occurred at a ask price level with size change < 0.</ul>

#### From trade data:
<ul>An event is classified as a <b>market buy</b> if the volume-weighted average execution price is equal to or above the bid-ask midpoint.</ul>
<ul>An event is classified as a <b>market sell</b> if the volume-weighted average execution price is below the mid-ask midpoint.</ul>
    
<b>Note:</b> Since only the Level-1 order book is to be modeled, the algorithm generally disregards any event that does not occur at the most competitive price level. This could be relaxed in the future if more order levels are to be modeled. 
    
The event tagging step is the final process in the data engineering pipeline, and the event data is passed onto the model. 

<a id = "sec_4"></a>
## IV. Model Construction

The model architecture itself is somewhat complex and employs both conventional and blackbox optimization procedures. 

The first optimization step revolves around determining the optimal mixing parameter $\alpha$ for the global and local model parameters. 

The second optimization step is performed to add a linear bias term to the simulation prediction, which aims to account for factors not incorporated into the simulation model itself. 



<a id = "sec_5"></a>
## V. Performance Analysis

The primary limitation of the model as it currently stands is that it is very computationally expensive. Each simulation currently takes around 40 milliseconds to execute (which really doesn't seem like a lot!). However, given that there are more than 400,000 target values in training and upwards of 150,000 in testing, a 40ms prediction time implies total execution time of over 4 hours even if just a <i>single</i> trial is conducted per simulation - and that doesn't even account for the additional time required to re-engineer the data!

As a result of the disappointingly slow execution speed, it has been difficult to evaluate the overall performance of the model. However, smaller scale evaluation has produced some encouraging results. 

<a id = "sec_6"></a>
## VI. Conclusion

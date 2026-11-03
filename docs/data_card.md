# Data Card: H&M Personalized Fashion Recommendations

## 1. Dataset Overview
- **Source**: Kaggle competition `h-and-m-personalized-fashion-recommendations`
- **Scope**: Tabular and text metadata only (`transactions_train.csv`, `customers.csv`, `articles.csv`). **No image data used.**
- **Dates**: 2018-09-20 to 2020-09-22 (~31.7M total transactions in raw source)
- **Active Window**: 12 months (2019-09-23 to 2020-09-22)

## 2. Sampling Configuration
- **Window Start Date**: `2019-09-23`
- **Window End Date**: `2020-09-22`
- **Sampling Strategy**: Customer-level sampling (keep 100% of historical transactions within window for sampled users)
- **Sample Fraction (`sample_frac`)**: `0.10` (10.0%)
- **Random Seed**: `42`
- **Sampling Formula**: `abs(hash(customer_id || seed)) % 1000000 < 100000`

## 3. Split Definitions (Leakage Contract)
- **Validation Week (V)**: `2020-09-16` to `2020-09-22` (Final metric reporting)
- **Ranker-Train Week (R)**: `2020-09-09` to `2020-09-15` (Ranker label generation)
- **Leakage Contract**: For any target week $T$, all candidate generation, feature aggregations, and user profiles use data strictly before $T_{\text{start}}$:
  $$\max(\text{feature\_source\_date}) < T_{\text{start}}$$

## 4. Empirical Dataset Statistics

| Dimension | Raw Dataset | 10% Sampled Parquet Pipeline | Notes |
| :--- | :--- | :--- | :--- |
| **Transactions** | ~31,788,324 | **1,489,401** | Saved in `artifacts/parquet/transactions_sample.parquet` (24.8 MB) |
| **Customers (Catalog)** | 1,371,980 | **99,338** | Sampled active customers in 12-month window |
| **Active Customers (12m)** | 994,320 | **99,338** | Sample captures exactly 9.99% of active window customers |
| **Articles (Catalog)** | 105,542 | **105,542** | Full catalog preserved with leading zero strings |
| **Active Articles in Sample** | - | **50,892** | 54,650 articles (51.8%) had zero sales in the 12m window |
| **Validation Week Buyers** | - | **6,837** | Active customers with $\ge 1$ purchase in Week V |
| **Validation Ground-Truth Pairs** | - | **20,988** | Positive (customer, article) validation targets |

## 5. Column Specifications & Quality Checks

### Transactions (`transactions_sample.parquet`)
- `t_dat`: Date (`DATE`, 2019-09-23 to 2020-09-22)
- `customer_id`: Hexadecimal string hash (`VARCHAR`)
- `customer_idx`: Integer index `[0, 99337]` (`INTEGER`)
- `article_id`: Strictly 10-digit zero-padded string (`VARCHAR`)
- `article_idx`: Integer index `[0, 105541]` (`INTEGER`)
- `price`: Scaled transaction price (`FLOAT`, min: 0.0001, median: 0.0254, p99: 0.0974, max: 0.5068)
- `sales_channel_id`: Channel indicator (`TINYINT`, 1 = In-store [28.1%], 2 = Online [71.9%])

### Customers (`customers_sample.parquet`)
- `customer_id`: Primary key string
- `customer_idx`: Integer key
- `FN`: 63.0% missing (imputed as 0)
- `Active`: 63.6% missing (imputed as 0)
- `club_member_status`: 0.28% missing (categories: `ACTIVE`, `PRE-CREATE`, `LEFT CLUB`)
- `fashion_news_frequency`: 1.30% missing (`NONE`, `Regularly`)
- `age`: 0.78% missing (`TINYINT`, min: 16, median: 32, max: 99; missing imputed by median)
- `postal_code`: Categorical hash string

### Articles (`articles.parquet`)
- `article_id`: Primary key string (10 characters)
- `product_code`: 7-digit style code string
- `prod_name`, `product_type_name`, `product_group_name`, `department_name`, `index_name`, `section_name`, `garment_group_name`, `detail_desc`: Textual and categorical metadata.

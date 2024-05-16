# Install

```bash
pip install -r requirements.txt
```

# Image explanation
## gpt test data
把gpt給的sample做簡單的實驗
- gpt_compare_MSE : 比較兩個特徵正規化前、兩個特徵正規化後、所有特徵無正規化的MSE
- gpt_norm_origin_plane : 顯示正規化後特徵的分布，包含平面與原始資料
- gpt_norm_test_compare_one : 顯示正規化後特徵的分布，包含平面與一筆預測的測試資料與原始資料的比對
- gpt_norm_test_compare : 顯示正規化後特徵的分布，包含平面與所有預測的測試資料與原始資料的比對
- gpt_origin_plane : 顯示正規化前特徵的分布，包含平面與原始資料
- gpt_test_compare_one : 顯示正規化前特徵的分布，包含平面與一筆預測的測試資料與原始資料的比對
- gpt_test_compare : 顯示正規化前特徵的分布，包含平面與所有預測的測試資料與原始資料的比對

## house test data
房屋價格預測的測試資料
根據上面的命名規則有一樣的意義

# Resources from kaggle
> https://www.kaggle.com/datasets/vikrishnan/boston-house-prices
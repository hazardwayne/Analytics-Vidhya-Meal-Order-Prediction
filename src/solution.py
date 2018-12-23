import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

# read data
train_df = pd.read_csv('../data/train.csv')
meal_df = pd.read_csv('../data/meal_info.csv')
fufil_df = pd.read_csv('../data/fulfilment_center_info.csv')
test_df = pd.read_csv('../data/test_QoiMO9B.csv')
sub_df = pd.read_csv('../data/sample_submission_hSlSoT6.csv')

# concat train and test dataframes
trn_size, tst_size = train_df.shape[0], test_df.shape[0]
df = pd.concat([train_df, test_df], axis=0, sort=False)

# join dataframes
df = pd.merge(df, meal_df, on='meal_id', how='left')
df = pd.merge(df, fufil_df, on='center_id', how='left')

# category encoding
df['category'] = df['category'].astype('category').cat.codes
df['cuisine'] = df['cuisine'].astype('category').cat.codes
df['city_code'] = df['city_code'].astype('category').cat.codes
df['center_type'] = df['center_type'].astype('category').cat.codes


# group function
def group_feature(df, col, feature, agg):
    feature_name = '_'.join(col) + '_{}_{}'.format(feature, agg)
    gp = df.groupby(col)[feature].agg(agg).reset_index().rename(columns={feature:feature_name})
    df = pd.merge(df, gp, on=col, how='left')
    
    df['{}_ratio'.format(feature_name)] = df[feature] / df[feature_name]
    df['{}_ratio'.format(feature_name)] = df['{}_ratio'.format(feature_name)].fillna(1.0)
#     df = df.drop(feature_name, axis=1)
    
    return df

# prices
cols = [['meal_id'], ['region_code', 'category', 'cuisine'], ['center_id'], 
        ['region_code'], ['center_id', 'meal_id'], ['region_code', 'meal_id']]
features = ['checkout_price', 'base_price']
for feature in features:
    for col in cols:
        df = group_feature(df, col, feature, 'mean')

# promotion
df['promotion'] = df['emailer_for_promotion'] + df['homepage_featured']

cols = [['meal_id'], ['center_id'], ['region_code'], ['region_code', 'category', 'cuisine']]
features = ['promotion']
for feature in features:
    for col in cols:
        df = group_feature(df, col, feature, 'sum')

# category comble features
cols = [['city_code'], ['center_id'], ['region_code']]
features = ['category', 'cuisine']
for feature in features:
    for col in cols:
        df = group_feature(df, col, feature, 'nunique')

cols = [['city_code'], ['region_code']]
features = ['center_type', 'meal_id']
for feature in features:
    for col in cols:
        df = group_feature(df, col, feature, 'nunique')
        

def history_feature(df, col, feature, agg):
    history_df = pd.DataFrame()
    feature_name = '_'.join(col) + '_{}'.format(feature)
    feature_name = '{}_history_{}'.format(feature_name, agg)
     
    for wk in range(1+df.week.max()+1):
        wk_df = df.loc[df.week<wk, :]
        
        history_gp = wk_df.groupby(col)[feature].agg(agg).reset_index() \
                          .rename(columns={feature:feature_name})
        
        history_gp['week'] = wk
        history_df = pd.concat([history_df, history_gp], axis=0)
        
    df = pd.merge(df, history_df, on=col+['week'], how='left')
    
    # fill with mean
    feature_mean_name = '{}_mean'.format(feature)
    mean_gp = df.groupby(col)[feature].agg(agg).reset_index().rename(columns={feature:feature_mean_name})
    df = pd.merge(df, mean_gp, on=col, how='left')
    df[feature_name] = df[feature_name].fillna(df[feature_mean_name])
    df = df.drop(feature_mean_name, axis=1)
    
    return df, feature_name

# target history
cols = [['center_id'], ['region_code', 'center_type', 'category'], ['city_code', 'cuisine'], 
        ['center_id', 'meal_id'], ['region_code', 'meal_id']]
features = ['num_orders']
feature_names = []
for feature in features:
    for col in cols:
        df, feature_name = history_feature(df, col, feature, 'mean')
        feature_names.append(feature_name)

# group comparing
combinations = []
for feature1 in feature_names:
    for feature2 in feature_names:
        if feature1 != feature2 and '{}_{}'.format(feature1, feature2) not in combinations and '{}_{}'.format(feature2, feature1) not in combinations:
            df['{}_{}_ratio'.format(feature1, feature2)] = df[feature1] / df[feature2]
            combinations.append('{}_{}'.format(feature1, feature2))


def diff_feature(df, col, feature):
    feature_name = '_'.join(col) + '_{}_diff'.format(feature)
    gp = df.groupby(col)[feature].mean()
    
    orig_df = gp.reset_index().rename(columns={feature: feature_name.replace('diff', 'orig')})
    # prevent overfitting
    if feature == 'num_orders':
        diff_df = gp.groupby(level=[i for i in range(len(col)-1)]).diff(1).shift(1).reset_index().rename(columns={feature: feature_name})
    else:    
        diff_df = gp.groupby(level=[i for i in range(len(col)-1)]).diff(1).reset_index().rename(columns={feature: feature_name})
    
    df = pd.merge(df, orig_df, on=col, how='left')
    df = pd.merge(df, diff_df, on=col, how='left')
    
    feature_discount_name = feature_name.replace('diff', 'discount')
    df[feature_discount_name] = df[feature_name] / df[feature_name.replace('diff', 'orig')]
    df[feature_discount_name] = df[feature_discount_name].apply(lambda x: 0.0 if x<0.1 else x)
    
    # fill na with mean
    feature_discount_mean_name = '{}_mean'.format(feature_discount_name)
    mean_col = [x for x in col if x != 'week']
    mean_df = df.groupby(mean_col)[feature_discount_name].mean().reset_index().rename(columns={feature_discount_name:feature_discount_mean_name})
    df = pd.merge(df, mean_df, on=mean_col, how='left')
    
    df[feature_discount_name] = df[feature_discount_name].fillna(df[feature_discount_mean_name])
    df = df.drop([feature_name, feature_name.replace('diff', 'orig'), feature_discount_mean_name], axis=1)
    
    return df

cols = [['center_id', 'category', 'week'], ['region_code', 'meal_id', 'week']]
features = ['checkout_price']
for feature in features:
    for col in cols:
        df = diff_feature(df, col, feature)

# save temp feature
# df.to_csv('../data/df_feature.csv', index=None)

# split dataset
train_da, valid_da, test_da = df.loc[df.week<136, :], df.loc[(df.week>=136)&(df.week<=145), :], df.loc[df.week>145, :]

ignorecols = ['id', 'num_orders', 'week']
features = [col for col in df.columns.tolist() if col not in ignorecols]

train_lgb = lgb.Dataset(train_da[features], train_da.num_orders.map(np.log1p))
valid_lgb = lgb.Dataset(valid_da[features], valid_da.num_orders.map(np.log1p))

params={
        'boosting':'gbdt',
        'objective':'regression_l2',
        'metric':'rmse',
        'learning_rate':0.1,
        'max_depth':6,
        'num_leaves':64,
        'bagging_fraction':0.7,
        'feature_fraction':0.7,
        'bagging_freq':4,
        'lambda_l2':3.0,
        'num_threads':4,
        'zero_as_missing':True,
        'seed':731
       }

lgb_train = lgb.train(params, 
                      train_set=train_lgb, 
                      num_boost_round=2000, 
                      valid_sets=[train_lgb, valid_lgb], 
                      valid_names=['train', 'valid'],
                      early_stopping_rounds=50, 
                      verbose_eval=30)

train_da_all = df.loc[df.week<146, :]
best_iteration = lgb_train.best_iteration

train_lgb_all = lgb.Dataset(train_da_all[features], train_da_all.num_orders.map(np.log1p))

lgb_train_all = lgb.train(params, 
                          train_set=train_lgb_all, 
                          num_boost_round=best_iteration)

predict_lgb = np.expm1(lgb_train_all.predict(test_da[features]))


#### xgboost
X_train, y_train = train_da_all[features], train_da_all.num_orders.map(np.log1p)
X_test = test_da[features]

best_iteration = 800 # has been tuned
xgb_clf = xgb.XGBRegressor(n_estimators=best_iteration, 
                           max_depth=8,
                           learning_rate=0.08,
                           subsample=0.75, 
                           colsample_bytree=0.75,
                           colsample_bylevel=0.75,
                           reg_lambda=1.0,
                           random_state=731)

xgb_clf.fit(X_train, y_train)

predict_xgb = np.expm1(xgb_clf.predict(X_test))


#### submission
sub_df['num_orders_lgb'] = predict_lgb
sub_df['num_orders_xgb'] = predict_xgb

# weighted average
sub_df['num_orders'] = (0.35 * sub_df['num_orders_lgb'] + 0.65 * sub_df['num_orders_xgb'])
sub_df[['id', 'num_orders']].to_csv('../sub/submission.csv', index=None)
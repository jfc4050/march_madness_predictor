import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

# quality of life function
def peek(df, n_cols=5):
   print('dataframe shape = {}'.format(df.shape))
   display(df.head(n_cols))


def get_X_y(df):
   # extract feature and target matrices from dataframe
   X = df.as_matrix(columns=df.columns.difference(['y']))
   y = df.as_matrix(columns=['y']).ravel()
   
   # normalize X: mean=0, SD=1
   X = StandardScaler().fit_transform(X)
      
   # sanity check
   print('{:<12} = {}'.format('X.shape', X.shape), '\n'
         '{:<12} = {}'.format('y.shape', y.shape))
         
   return X, y


def pca_visualize(x, y):
   # normalize features
   scaler = StandardScaler()
   x = scaler.fit_transform(x)
   
   # fit and project
   pca = PCA(n_components=2, svd_solver='randomized')
   x_pca = pca.fit_transform(x)
      
   # place projections into dataframe for plotting
   pca_df = pd.DataFrame(np.column_stack((y, x_pca)),
                         columns=['y', '1st principal component', '2nd principal component']).sample(500)
   pca_df['color']= np.where(pca_df['y'] == 1, "#9b59b6", "#3498db")
      
   fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
      
   sns.regplot(data=pca_df,
               x='1st principal component',
               y='2nd principal component',
               fit_reg=False,
               scatter_kws={'facecolors':pca_df['color']},
               ax=ax1)
      
   pca = PCA(svd_solver='randomized').fit(x)
   ax2.plot(np.cumsum(pca.explained_variance_ratio_))
   ax2.set_xlim([0, None])
   ax2.set_ylim([0, 1.2])
   ax2.set_xlabel('n_components')
   ax2.set_ylabel('cumulative explained variance')


def tsne_visualize(x, y):
   # normalize features
   x = StandardScaler().fit_transform(x)
   
   # for noise suppression/ reducing computational expense
   if x.shape[1] > 50:
      x = PCA(50).fit_transform(x)
         
   # fit and project
   x_tsne = TSNE(2).fit_transform(x)
         
   tsne_df = pd.DataFrame(np.column_stack((y, x_tsne)), columns=['y', '1st tsne component', '2nd tsne component']).sample(500)
   tsne_df['color'] = np.where(tsne_df['y'] == 1, "#9b59b6", "#3498db")
         
         
   sns.regplot(data=tsne_df, x='1st tsne component', y='2nd tsne component', fit_reg=False, scatter_kws={'facecolors':tsne_df['color']})


def plot_learning_curve(train_sizes, train_scores, valid_scores):
   
   # get mean and standard deviation values for training and validation scores
   train_scores_mean = np.mean(train_scores, axis=1)
   train_scores_std  = np.std(train_scores, axis=1)
   valid_scores_mean = np.mean(valid_scores, axis=1)
   valid_scores_std  = np.std(valid_scores, axis=1)
   
   fig, ax = plt.subplots();
      
   # plot training score and standard deviation
   ax.plot(train_sizes, train_scores_mean, color='b', label='Training Score')
   ax.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, color='b', alpha=0.1)
      
   # plot validation score and standard deviation
   ax.plot(train_sizes, valid_scores_mean, color='r', label='Cross-Validation Score')
   ax.fill_between(train_sizes, valid_scores_mean-valid_scores_std, valid_scores_mean+valid_scores_std, color='r', alpha=0.1)
      
   ax.legend(loc='best');
      
   return ax


def loss_accuracy_report(model, data_tr, data_val):
   # unpack training and validation data
   X_tr, y_tr = data_tr
   X_val, y_val = data_val
         
   # train new model
   reset_model = clone(model)
   reset_model.fit(X_tr, y_tr)

   # compute accuracy scores
   accuracy_tr  = accuracy_score(y_tr , reset_model.predict(X_tr))
   accuracy_val = accuracy_score(y_val, reset_model.predict(X_val))
         
   # compute log losses
   loss_tr  = log_loss(y_tr, reset_model.predict_proba(X_tr))
   loss_val = log_loss(y_val, reset_model.predict_proba(X_val))
         
   # plot report as table
   cell_text = [[loss_tr, accuracy_tr],
                [loss_val, accuracy_val]]
            
   fig, ax = plt.subplots(1, 1, figsize=(10, 1))
   ax.axis('off')
   ax.table(cellText=cell_text,
            rowLabels=['train', 'test'],
            colLabels=['log_loss', 'accuracy'],
            loc='center')

   return ax

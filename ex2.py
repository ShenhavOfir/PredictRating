import abc
from typing import Tuple
import pandas as pd
import numpy as np
import datetime as dt
from numpy.linalg import norm


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        sum = 0
        for i in range(true_ratings.shape[0]):
            predicted_rating = self.predict(true_ratings['user'][i], true_ratings['item'][i], true_ratings['timestamp'][i])
            sum += (true_ratings['rating'][i] - predicted_rating) ** 2
        return np.sqrt(sum / true_ratings.shape[0])


class BaselineRecommender(Recommender):
    def __init__(self, ratings: pd.DataFrame):
        super().__init__(ratings)

    def initialize_predictor(self, ratings: pd.DataFrame):
        b_u = {}
        b_i = {}
        self.r_avg = np.mean(ratings['rating'])
        df_users = ratings.groupby(['user'])['rating'].mean().reset_index()
        users_list = df_users['user'].values.tolist()
        user_avg_rating_list = df_users['rating'].values.tolist()
        for i in users_list:
            b_u[i] = user_avg_rating_list[int(i)] - self.r_avg
        df_items = ratings.groupby(['item']).mean().reset_index()
        items_list = (df_items['item'].values.tolist())
        item_avg_rating_list = (df_items['rating']).values.tolist()
        for i in items_list:
            b_i[i] = item_avg_rating_list[int(i)] - self.r_avg
        self.b_u = b_u
        self.b_i = b_i

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        predicted = self.r_avg + self.b_u[user] + self.b_i[item]
        if predicted > 5:
            return 5
        elif predicted < 0.5:
            return 0.5
        else:
            return predicted


class NeighborhoodRecommender(Recommender):
    def __init__(self, ratings: pd.DataFrame):
        u = ratings['user'].nunique()
        matrix = np.empty((u + 1, u + 1))
        matrix[:] = -2
        self.matrix_user_similarity = matrix
        self.u = u
        self.avg_ratings = 0
        self.b_user = {}
        self.b_item = {}
        self.rating_after_avg = pd.DataFrame()
        self.rater_of_item = {}
        self.rate_of_user = {}
        super().__init__(ratings)

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.avg_ratings = ratings["rating"].mean()

        user_avg_ratings_list = ratings.groupby(['user'])['rating'].mean().reset_index() - self.avg_ratings
        self.b_user = user_avg_ratings_list.to_dict()['rating']

        df_items = ratings.groupby(['item'])['rating'].mean().reset_index() - self.avg_ratings
        self.b_item = df_items.to_dict()['rating']

        self.rating_after_avg = ratings[["user", "item", "rating"]]
        self.rating_after_avg["new_rating"] = self.rating_after_avg["rating"] - self.avg_ratings


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        upper = 0
        lower = 0
        corr = []
        # looking only at users that ranked this specific item
        if item not in self.rater_of_item:
            raters = self.rating_after_avg["item"].values == item
            self.rater_of_item[item] = self.rating_after_avg[raters]["user"].unique().astype(int)

        user_df = self.rater_of_item[item]

        # computing corr only for users who ranked the specific item
        for x in user_df:
            # meaning the correlation between this user to other user was not calculated yet
            if user == x:
                self.matrix_user_similarity[int(user)][int(user)] = 1

            else:
                corr.append(self.user_similarity(user, x))

        neighbors_list = np.array(corr)
        neighbors_list = (abs(neighbors_list)).argsort()[-3:][::-1]
        for k in neighbors_list:
            rate = self.rate_of_user[user_df[int(k)]][item]
            upper += self.matrix_user_similarity[int(user)][user_df[int(k)]] * rate
            lower += abs(corr[k])

        predicted = self.avg_ratings + self.b_user[user] + self.b_item[item] + upper / lower
        if predicted > 5:
            return 5
        elif predicted < 0.5:
            return 0.5
        else:
            return predicted


    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        corr = 0
        if self.matrix_user_similarity[int(user1)][int(user2)] != -2:
            return self.matrix_user_similarity[int(user1)][int(user2)]

        if user1 not in self.rate_of_user:
            user1_ratings = self.rating_after_avg[self.rating_after_avg['user'] == user1][['item', 'new_rating']]
            self.rate_of_user[user1] = user1_ratings.set_index("item")["new_rating"].to_dict()

        user1_ratings = self.rate_of_user[user1]
        # user1_ratings = pd.DataFrame(list(user1_ratings.items()), columns=['item', 'rating1'])
        # user1_ratings = pd.DataFrame(user1_ratings, columns=['Column_A', 'Column_B', 'Column_C'])
        # user1_ratings = user1_ratings.rename(columns={'rating': 'rating1'})

        if user2 not in self.rate_of_user:
            user2_ratings = self.rating_after_avg[self.rating_after_avg['user'] == user2][['item', 'new_rating']]
            self.rate_of_user[user2] = user2_ratings.set_index("item")["new_rating"].to_dict()

        user2_ratings = self.rate_of_user[user2]
        # user2_ratings = pd.DataFrame(list(user2_ratings.items()), columns=['item', 'rating2'])

        user_1_rate_intersection = []
        user_2_rate_intersection = []
        for key in user1_ratings.keys():
            if key in user2_ratings.keys():
                user_1_rate_intersection.append(user1_ratings[key])
                user_2_rate_intersection.append(user2_ratings[key])

        # merge_ratings = pd.merge(user1_ratings, user2_ratings, how='inner', on=['item'])
        # merge_ratings['sum'] = merge_ratings['rating1'] * merge_ratings['rating2']
        upper = np.dot(user_1_rate_intersection, user_2_rate_intersection)
        lower = norm(user_1_rate_intersection) * norm(user_2_rate_intersection)
        if lower == 0 or lower is None:
            corr = 0
        else:
            corr = upper / lower

        self.matrix_user_similarity[int(user1)][int(user2)] = corr
        self.matrix_user_similarity[int(user2)][int(user1)] = corr
        return float(corr)


class LSRecommender(Recommender):

    def __init__(self, ratings: pd.DataFrame):
        self.u = ratings['user'].nunique()
        super().__init__(ratings)

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.ratings = ratings
        self.r_avg = np.mean(ratings['rating'])
        self.tuple = self.solve_ls()

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        b = self.tuple[1]
        b_u = b[int(user)]
        b_i = b[int(self.u + item)]
        b_d = b[-3]
        b_n = b[-2]
        b_w = b[-1]
        dt_object = dt.datetime.fromtimestamp(timestamp)
        if (6 <= dt_object.hour <= 18):
            if (dt_object.strftime("%A") == 'Friday' or dt_object.strftime("%A") == 'Saturday'):
                prediction = self.r_avg + b_u + b_i + b_d + b_w
            else:
                prediction = self.r_avg + b_u + b_i + b_d

        else:
            if (dt_object.strftime("%A") == 'Friday' or dt_object.strftime("%A") == 'Saturday'):
                prediction = self.r_avg + b_u + b_i + b_n + b_w
            else:
                prediction = self.r_avg + b_u + b_i + b_n
        return prediction

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        u = self.u
        ratings = self.ratings
        ratings['y'] = ratings['rating'] - self.r_avg
        y = ratings['y'].to_numpy()
        i = ratings['item'].nunique()
        rows = ratings.shape[0]
        columns = u + i + 3
        x = np.ndarray(shape=(rows, columns))
        for j in range(rows):
            user_column = int(ratings['user'][j])
            x[j][user_column] = 1
            item_column = u + int(ratings['item'][j])
            x[j][item_column] = 1
            dt_object = dt.datetime.fromtimestamp(ratings['timestamp'][j])
            if (6 <= dt_object.hour <= 18):
                x[j][-3] = 1
            else:
                x[j][-2] = 1
            if (dt_object.strftime("%A") == 'Friday' or dt_object.strftime("%A") == 'Saturday'):
                x[j][-1] = 1
        beta, a, b, c = np.linalg.lstsq(x, y, rcond=None)
        tuple = (x, beta, y)
        return tuple

class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.df = ratings
        n_users = self.df['user'].unique().shape[0]
        n_items = self.df['item'].unique().shape[0]
        ratings = np.zeros((n_users, n_items))
        for row in ratings.itertuples(index=False):
            ratings[row.user_id - 1, row.item_id - 1] = row.rating

        # compute the non-zero elements in the rating matrix
        matrix_size = np.prod(ratings.shape)
        interaction = np.flatnonzero(ratings).shape[0]
        sparsity = 100 * (interaction / matrix_size)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

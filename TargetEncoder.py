from numbers import Integral, Real

import numpy as np

from ..base import OneToOneFeatureMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import type_of_target
from ..utils.validation import (
    _check_feature_names_in,
    _check_y,
    check_consistent_length,
    check_is_fitted,
)
from ._encoders import _BaseEncoder
from ._target_encoder_fast import _fit_encoding_fast, _fit_encoding_fast_auto_smooth


class TargetEncoder(OneToOneFeatureMixin, _BaseEncoder):

    _parameter_constraints: dict = {
        "categories": [StrOptions({"auto"}), list],
        "target_type": [StrOptions({"auto", "continuous", "binary", "multiclass"})],
        "smooth": [StrOptions({"auto"}), Interval(Real, 0, None, closed="left")],
        "cv": [Interval(Integral, 2, None, closed="left")],
        "shuffle": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        categories="auto",
        target_type="auto",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=None,
    ):
        self.categories = categories
        self.smooth = smooth
        self.target_type = target_type
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
    
        self._fit_encodings_all(X, y)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y):
       
        from ..model_selection import KFold, StratifiedKFold  # avoid circular import

        X_ordinal, X_known_mask, y_encoded, n_categories = self._fit_encodings_all(X, y)

        # The cv splitter is voluntarily restricted to *KFold to enforce non
        # overlapping validation folds, otherwise the fit_transform output will
        # not be well-specified.
        if self.target_type_ == "continuous":
            cv = KFold(self.cv, shuffle=self.shuffle, random_state=self.random_state)
        else:
            cv = StratifiedKFold(
                self.cv, shuffle=self.shuffle, random_state=self.random_state
            )

        # If 'multiclass' multiply axis=1 by num classes else keep shape the same
        if self.target_type_ == "multiclass":
            X_out = np.empty(
                (X_ordinal.shape[0], X_ordinal.shape[1] * len(self.classes_)),
                dtype=np.float64,
            )
        else:
            X_out = np.empty_like(X_ordinal, dtype=np.float64)

        for train_idx, test_idx in cv.split(X, y):
            X_train, y_train = X_ordinal[train_idx, :], y_encoded[train_idx]
            y_train_mean = np.mean(y_train, axis=0)

            if self.target_type_ == "multiclass":
                encodings = self._fit_encoding_multiclass(
                    X_train,
                    y_train,
                    n_categories,
                    y_train_mean,
                )
            else:
                encodings = self._fit_encoding_binary_or_continuous(
                    X_train,
                    y_train,
                    n_categories,
                    y_train_mean,
                )
            self._transform_X_ordinal(
                X_out,
                X_ordinal,
                ~X_known_mask,
                test_idx,
                encodings,
                y_train_mean,
            )
        return X_out

    def transform(self, X):
     
        X_ordinal, X_known_mask = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )

        # If 'multiclass' multiply axis=1 by num of classes else keep shape the same
        if self.target_type_ == "multiclass":
            X_out = np.empty(
                (X_ordinal.shape[0], X_ordinal.shape[1] * len(self.classes_)),
                dtype=np.float64,
            )
        else:
            X_out = np.empty_like(X_ordinal, dtype=np.float64)

        self._transform_X_ordinal(
            X_out,
            X_ordinal,
            ~X_known_mask,
            slice(None),
            self.encodings_,
            self.target_mean_,
        )
        return X_out

    def _fit_encodings_all(self, X, y):
        """Fit a target encoding with all the data."""
        # avoid circular import
        from ..preprocessing import (
            LabelBinarizer,
            LabelEncoder,
        )

        check_consistent_length(X, y)
        self._fit(X, handle_unknown="ignore", force_all_finite="allow-nan")

        if self.target_type == "auto":
            accepted_target_types = ("binary", "multiclass", "continuous")
            inferred_type_of_target = type_of_target(y, input_name="y")
            if inferred_type_of_target not in accepted_target_types:
                raise ValueError(
                    "Unknown label type: Target type was inferred to be "
                    f"{inferred_type_of_target!r}. Only {accepted_target_types} are "
                    "supported."
                )
            self.target_type_ = inferred_type_of_target
        else:
            self.target_type_ = self.target_type

        self.classes_ = None
        if self.target_type_ == "binary":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            self.classes_ = label_encoder.classes_
        elif self.target_type_ == "multiclass":
            label_binarizer = LabelBinarizer()
            y = label_binarizer.fit_transform(y)
            self.classes_ = label_binarizer.classes_
        else:  # continuous
            y = _check_y(y, y_numeric=True, estimator=self)

        self.target_mean_ = np.mean(y, axis=0)

        X_ordinal, X_known_mask = self._transform(
            X, handle_unknown="ignore", force_all_finite="allow-nan"
        )
        n_categories = np.fromiter(
            (len(category_for_feature) for category_for_feature in self.categories_),
            dtype=np.int64,
            count=len(self.categories_),
        )
        if self.target_type_ == "multiclass":
            encodings = self._fit_encoding_multiclass(
                X_ordinal,
                y,
                n_categories,
                self.target_mean_,
            )
        else:
            encodings = self._fit_encoding_binary_or_continuous(
                X_ordinal,
                y,
                n_categories,
                self.target_mean_,
            )
        self.encodings_ = encodings

        return X_ordinal, X_known_mask, y, n_categories

    def _fit_encoding_binary_or_continuous(
        self, X_ordinal, y, n_categories, target_mean
    ):
        """Learn target encodings."""
        if self.smooth == "auto":
            y_variance = np.var(y)
            encodings = _fit_encoding_fast_auto_smooth(
                X_ordinal,
                y,
                n_categories,
                target_mean,
                y_variance,
            )
        else:
            encodings = _fit_encoding_fast(
                X_ordinal,
                y,
                n_categories,
                self.smooth,
                target_mean,
            )
        return encodings

    def _fit_encoding_multiclass(self, X_ordinal, y, n_categories, target_mean):

        n_features = self.n_features_in_
        n_classes = len(self.classes_)

        encodings = []
        for i in range(n_classes):
            y_class = y[:, i]
            encoding = self._fit_encoding_binary_or_continuous(
                X_ordinal,
                y_class,
                n_categories,
                target_mean[i],
            )
            encodings.extend(encoding)

        reorder_index = (
            idx
            for start in range(n_features)
            for idx in range(start, (n_classes * n_features), n_features)
        )
        return [encodings[idx] for idx in reorder_index]

    def _transform_X_ordinal(
        self,
        X_out,
        X_ordinal,
        X_unknown_mask,
        row_indices,
        encodings,
        target_mean,
    ):
  
        if self.target_type_ == "multiclass":
            n_classes = len(self.classes_)
            for e_idx, encoding in enumerate(encodings):
                # Repeat feature indices by n_classes
                feat_idx = e_idx // n_classes
                # Cycle through each class
                mean_idx = e_idx % n_classes
                X_out[row_indices, e_idx] = encoding[X_ordinal[row_indices, feat_idx]]
                X_out[X_unknown_mask[:, feat_idx], e_idx] = target_mean[mean_idx]
        else:
            for e_idx, encoding in enumerate(encodings):
                X_out[row_indices, e_idx] = encoding[X_ordinal[row_indices, e_idx]]
                X_out[X_unknown_mask[:, e_idx], e_idx] = target_mean

    def get_feature_names_out(self, input_features=None):
    
        check_is_fitted(self, "n_features_in_")
        feature_names = _check_feature_names_in(self, input_features)
        if self.target_type_ == "multiclass":
            feature_names = [
                f"{feature_name}_{class_name}"
                for feature_name in feature_names
                for class_name in self.classes_
            ]
            return np.asarray(feature_names, dtype=object)
        else:
            return feature_names

    def _more_tags(self):
        return {
            "requires_y": True,
        }
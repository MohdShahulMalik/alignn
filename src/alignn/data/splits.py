from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def create_split_frames(
    frame: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, pd.DataFrame]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    if frame.empty:
        raise ValueError("Cannot split an empty dataframe.")

    train_frame, temp_frame = train_test_split(
        frame,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        shuffle=True,
    )
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        shuffle=True,
    )

    return {
        "train": train_frame.sort_values("jid").reset_index(drop=True),
        "val": val_frame.sort_values("jid").reset_index(drop=True),
        "test": test_frame.sort_values("jid").reset_index(drop=True),
    }

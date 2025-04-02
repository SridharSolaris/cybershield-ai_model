
# # working 
# import pandas as pd
# import numpy as np
# from faker import Faker
# from sklearn.preprocessing import OrdinalEncoder
# import joblib

# fake = Faker()

# def generate_synthetic_data(n_samples=10000):
#     # Ensure an equal number of abusive and non-abusive samples
#     n_abusive = n_samples // 2
#     n_non_abusive = n_samples - n_abusive

#     data = {
#         "countryCode": [fake.country_code() for _ in range(n_samples)],
#         "usageType": [
#             fake.random_element(
#                 elements=("Data Center/Web Hosting/Transit", "ISP", "Business", "Residential", "Fixed Line ISP")
#             )
#             for _ in range(n_samples)
#         ],
#         "domain": [fake.domain_name() for _ in range(n_samples)],
#         "isp": [fake.company() for _ in range(n_samples)],
#         "totalReports": np.concatenate((np.random.randint(10, 2000, n_abusive), np.random.randint(0, 10, n_non_abusive))),
#         "numDistinctUsers": np.concatenate((np.random.randint(10, 1000, n_abusive), np.random.randint(0, 10, n_non_abusive))),
#         "lastReportedAt": [fake.date_time_this_year().timestamp() for _ in range(n_samples)],
#         "isWhitelisted": np.random.choice([True, False], n_samples),
#         "isTor": np.random.choice([True, False], n_samples),
#     }

#     # Calculate abuseConfidenceScore using a weighted formula
#     def calculate_score(row):
#         score = (
#             0.4 * (row["totalReports"] / 2000) +
#             0.3 * (row["numDistinctUsers"] / 1000)
#         )
#         if row["isWhitelisted"]:
#             score *=0.1  # Reduce score for whitelisted entries
#         if row["isTor"]:
#             score += 0.2  # Increase score for Tor users
#         return min(100, max(0, score * 100))  # Ensure score is between 0 and 100

#     df = pd.DataFrame(data)
#     df["abuseConfidenceScore"] = df.apply(calculate_score, axis=1).astype(int)

#     # Convert the target into binary classes (1 if score > 50, else 0)
#     df["abuseConfidenceClass"] = np.concatenate((np.ones(n_abusive), np.zeros(n_non_abusive)))

#     # Save the raw data
#     df.to_csv("synthetic_data.csv", index=False)

#     # Encode categorical variables using OrdinalEncoder with handle_unknown='use_encoded_value'
#     encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#     encoded_categorical = encoder.fit_transform(df[["countryCode", "usageType", "domain", "isp"]])
#     encoded_df = pd.DataFrame(
#         encoded_categorical, columns=["countryCode", "usageType", "domain", "isp"]
#     )

#     # Concatenate encoded features with the rest of the DataFrame
#     df_encoded = pd.concat(
#         [df.drop(["countryCode", "usageType", "domain", "isp"], axis=1), encoded_df], axis=1
#     )

#     return df_encoded, encoder

# # Generate the synthetic data
# df_encoded, encoder = generate_synthetic_data()
# df_encoded.to_csv("synthetic_data_encoded.csv", index=False)
# joblib.dump(encoder, "encoder.pkl")
# print("Synthetic dataset generated and saved to 'synthetic_data.csv'")
# print("Encoded dataset saved to 'synthetic_data_encoded.csv'")
# print("Encoder saved to 'encoder.pkl'")


#!/usr/bin/env python
# import pandas as pd
# import numpy as np
# from faker import Faker
# from sklearn.preprocessing import OrdinalEncoder
# import joblib
# from joblib import Parallel, delayed
# import random

# # Set random seeds for reproducibility
# np.random.seed(42)
# random.seed(42)

# fake = Faker()
# Faker.seed(42)

# def generate_synthetic_data(n_samples=10000):
#     """
#     Generate synthetic data for risk scoring.

#     Notes:
#       - The features countryCode, usageType, domain, isp are generated using Faker.
#       - Numeric features (totalReports, numDistinctUsers) are generated separately so that
#         the first half of the samples are more “abusive” (with higher counts) and the second
#         half less abusive.
#       - abuseConfidenceScore is computed using a weighted formula.
#       - abuseConfidenceClass is manually set to 1 (abusive) for the first half of the samples
#         and 0 for the remainder to force balance. (This may not exactly match the computed score.)
#     """
#     # Ensure an equal number of abusive and non-abusive samples
#     n_abusive = n_samples // 2
#     n_non_abusive = n_samples - n_abusive

#     data = {
#         "countryCode": Parallel(n_jobs=-1)(delayed(fake.country_code)() for _ in range(n_samples)),
#         "usageType": Parallel(n_jobs=-1)(
#             delayed(fake.random_element)(elements=(
#                 "Data Center/Web Hosting/Transit", "ISP", "Business", "Residential", "Fixed Line ISP"
#             )) for _ in range(n_samples)
#         ),
#         "domain": Parallel(n_jobs=-1)(delayed(fake.domain_name)() for _ in range(n_samples)),
#         "isp": Parallel(n_jobs=-1)(delayed(fake.company)() for _ in range(n_samples)),
#         # For abusive samples, generate higher report counts; for non-abusive, very low counts.
#         "totalReports": np.concatenate((
#             np.random.randint(10, 2000, n_abusive),
#             np.random.randint(0, 10, n_non_abusive)
#         )),
#         "numDistinctUsers": np.concatenate((
#             np.random.randint(10, 1000, n_abusive),
#             np.random.randint(0, 10, n_non_abusive)
#         )),
#         "lastReportedAt": Parallel(n_jobs=-1)(delayed(fake.date_time_this_year().timestamp)() for _ in range(n_samples)),
#         "isWhitelisted": np.random.choice([True, False], n_samples),
#         "isTor": np.random.choice([True, False], n_samples),
#     }

#     def calculate_score(row):
#         score = (
#             0.4 * (row["totalReports"] / 2000) +
#             0.3 * (row["numDistinctUsers"] / 1000)
#         )
#         if row["isWhitelisted"]:
#             score *= 0.1  # Reduce score for whitelisted entries
#         if row["isTor"]:
#             score += 0.2  # Increase score for Tor users
#         return min(100, max(0, score * 100))  # Ensure score is between 0 and 100

#     df = pd.DataFrame(data)
#     df["abuseConfidenceScore"] = df.apply(calculate_score, axis=1).astype(int)

#     # Manually assign classes to enforce balance.
#     # Note: This may not exactly match a threshold on abuseConfidenceScore.
#     df["abuseConfidenceClass"] = np.concatenate((np.ones(n_abusive), np.zeros(n_non_abusive)))

#     # Save the raw data for reference.
#     df.to_csv("synthetic_data.csv", index=False)

#     # Encode categorical variables using OrdinalEncoder
#     encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
#     encoded_categorical = encoder.fit_transform(df[["countryCode", "usageType", "domain", "isp"]])
#     encoded_df = pd.DataFrame(
#         encoded_categorical, columns=["countryCode", "usageType", "domain", "isp"]
#     )

#     # Concatenate encoded features with the rest of the DataFrame.
#     # The final column order (for training) will be:
#     # [totalReports, numDistinctUsers, lastReportedAt, isWhitelisted, isTor,
#     #  countryCode, usageType, domain, isp, abuseConfidenceScore, abuseConfidenceClass]
#     df_encoded = pd.concat(
#         [df.drop(["countryCode", "usageType", "domain", "isp"], axis=1), encoded_df],
#         axis=1
#     )

#     return df_encoded, encoder

# if __name__ == '__main__':
#     df_encoded, encoder = generate_synthetic_data()
#     df_encoded.to_csv("synthetic_data_encoded.csv", index=False)
#     joblib.dump(encoder, "encoder.pkl")
#     print("Synthetic dataset generated and saved to 'synthetic_data.csv'")
#     print("Encoded dataset saved to 'synthetic_data_encoded.csv'")
#     print("Encoder saved to 'encoder.pkl'")



import os
import joblib
import pandas as pd
import numpy as np
from faker import Faker
from sklearn.preprocessing import OrdinalEncoder
from joblib import Parallel, delayed
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

fake = Faker()
Faker.seed(42)

def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic data for risk scoring.
    """
    # Ensure an equal number of abusive and non-abusive samples
    n_abusive = n_samples // 2
    n_non_abusive = n_samples - n_abusive

    data = {
        "countryCode": Parallel(n_jobs=-1)(delayed(fake.country_code)() for _ in range(n_samples)),
        "usageType": Parallel(n_jobs=-1)(
            delayed(fake.random_element)(elements=(
                "Data Center/Web Hosting/Transit", "ISP", "Business", "Residential", "Fixed Line ISP"
            )) for _ in range(n_samples)
        ),
        "domain": Parallel(n_jobs=-1)(delayed(fake.domain_name)() for _ in range(n_samples)),
        "isp": Parallel(n_jobs=-1)(delayed(fake.company)() for _ in range(n_samples)),
        # For abusive samples, generate higher report counts; for non-abusive, very low counts.
        "totalReports": np.concatenate((
            np.random.randint(10, 2000, n_abusive),
            np.random.randint(0, 10, n_non_abusive)
        )),
        "numDistinctUsers": np.concatenate((
            np.random.randint(10, 1000, n_abusive),
            np.random.randint(0, 10, n_non_abusive)
        )),
        "lastReportedAt": Parallel(n_jobs=-1)(delayed(fake.date_time_this_year().timestamp)() for _ in range(n_samples)),
        "isWhitelisted": np.random.choice([True, False], n_samples),
        "isTor": np.random.choice([True, False], n_samples),
    }

    def calculate_score(row):
        score = (
            0.4 * (row["totalReports"] / 2000) +
            0.3 * (row["numDistinctUsers"] / 1000)
        )
        if row["isWhitelisted"]:
            score *= 0.1  # Reduce score for whitelisted entries
        if row["isTor"]:
            score += 0.2  # Increase score for Tor users
        return min(100, max(0, score * 100))  # Ensure score is between 0 and 100

    df = pd.DataFrame(data)
    df["abuseConfidenceScore"] = df.apply(calculate_score, axis=1).astype(int)

    # Manually assign classes to enforce balance.
    df["abuseConfidenceClass"] = np.concatenate((np.ones(n_abusive), np.zeros(n_non_abusive)))

    # Save the raw data for reference.
    df.to_csv("synthetic_data.csv", index=False)

    # Encode categorical variables using OrdinalEncoder.
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoded_categorical = encoder.fit_transform(df[["countryCode", "usageType", "domain", "isp"]])
    encoded_df = pd.DataFrame(
        encoded_categorical, columns=["countryCode", "usageType", "domain", "isp"]
    )

    # Concatenate encoded features with the rest of the DataFrame.
    df_encoded = pd.concat(
        [df.drop(["countryCode", "usageType", "domain", "isp"], axis=1), encoded_df],
        axis=1
    )

    return df_encoded, encoder

if __name__ == '__main__':
    df_encoded, encoder = generate_synthetic_data()
    df_encoded.to_csv("synthetic_data_encoded.csv", index=False)
    joblib.dump(encoder, "encoder.pkl")
    print("Synthetic dataset generated and saved to 'synthetic_data.csv'")
    print("Encoded dataset saved to 'synthetic_data_encoded.csv'")
    print("Encoder saved to 'encoder.pkl'")

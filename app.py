from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from functools import reduce
from rapidfuzz import fuzz
from dateutil.parser import parse
import argparse

mean_cols = udf(
    lambda array: int(reduce(lambda x, y: x + y, array) / len(array)), IntegerType()
)


def fuzzy_match(a: str, b: str) -> int:
    """
    This function is used for fuzzy matching the string types ,
    if both are nulls or empty strings , ratio will be 100 so in order to
    avoid that we have the below if conditions
    """
    if a is None or b is None:
        return 0
    elif len(a) < 2 or len(b) < 2:
        return 0
    else:
        return int(fuzz.token_sort_ratio(a, b))


def fuzzy_match_dates(a: str, b: str) -> int:
    """
    This function is used to fuzzy match the dates
    some dates are ignored which are not in a valid format for example 00/00/1960
    """
    a_data = []
    b_data = []
    if a is None or b is None:
        return 0
    elif len(a) < 2 or len(b) < 2:
        return 0
    else:
        a = a.split(",")
        b = b.split(",")
        for v in a:
            try:
                a_data.append(parse(v, fuzzy=True))
            except Exception as e:
                pass
        for u in b:
            try:
                b_data.append(parse(u, fuzzy=True))
            except Exception as e:
                pass
        return int(
            fuzz.token_set_ratio(
                ",".join(list(map(lambda x: str(x), a_data))),
                ",".join(list(map(lambda x: str(x), b_data))),
            )
        )


def main(threshold: int, ofac_path: str, uk_path: str, output: str):
    us = spark.read.option("compression", "gzip").json(ofac_path)
    uk = spark.read.option("compression", "gzip").json(uk_path)
    uk = (
        uk.withColumn("uk_countries", concat_ws(",", col("addresses.country")))
        .withColumn("uk_postal", concat_ws(",", col("addresses.postal_code")))
        .withColumn("uk_address", concat_ws(",", col("addresses.value")))
        .withColumn("uk_aliases", concat_ws(",", col("aliases.value")))
        .withColumn("uk_id_number", concat_ws(",", col("id_numbers.value")))
        .withColumn("uk_nationality", concat_ws(",", col("nationality")))
        .withColumn("uk_dob", concat_ws(",", col("reported_dates_of_birth")))
        .select(
            col("name").alias("uk_name"),
            col("id").alias("uk_id"),
            col("uk_address"),
            col("uk_postal"),
            col("uk_countries"),
            col("uk_aliases"),
            col("uk_id_number"),
            col("uk_nationality"),
            col("place_of_birth").alias("uk_place_of_birth"),
            col("uk_dob"),
        )
    )

    us = (
        us.withColumn("us_countries", concat_ws(",", col("addresses.country")))
        .withColumn("us_postal", concat_ws(",", col("addresses.postal_code")))
        .withColumn("us_address", concat_ws(",", col("addresses.value")))
        .withColumn("us_aliases", concat_ws(",", col("aliases.value")))
        .withColumn("us_id_number", concat_ws(",", col("id_numbers.value")))
        .withColumn("us_nationality", concat_ws(",", col("nationality")))
        .withColumn("us_dob", concat_ws(",", col("reported_dates_of_birth")))
        .select(
            col("name").alias("us_name"),
            col("id").alias("ofac_id"),
            col("us_address"),
            col("us_postal"),
            col("us_countries"),
            col("us_aliases"),
            col("us_id_number"),
            col("us_nationality"),
            col("place_of_birth").alias("us_place_of_birth"),
            col("us_dob"),
        )
    )
    Individual_data = us.filter(col("type") == "Individual").crossJoin(
        uk.filter(col("type") == "Individual")
    )
    Entity_data = us.filter(col("type") == "Entity").crossJoin(
        uk.filter(col("type") == "Entity")
    )

    data = Individual_data.union(Entity_data)
    fuzzy_match_udf = udf(lambda a, b: fuzzy_match(a, b), IntegerType())
    dates_udf = udf(lambda x, y: fuzzy_match_dates(x, y), IntegerType())
    cols_req = [
        "name_match",
        "address_match",
        "countries_match",
        "aliases_match",
        "id_number_match",
        "nationality_match",
        "pob_match",
        "dates_match"
    ]
    matches = (
        data.withColumn("name_match", fuzzy_match_udf(col("uk_name"), col("us_name")))
        .withColumn(
            "address_match", fuzzy_match_udf(col("uk_address"), col("us_address"))
        )
        .withColumn(
            "countries_match", fuzzy_match_udf(col("uk_countries"), col("us_countries"))
        )
        .withColumn(
            "aliases_match", fuzzy_match_udf(col("uk_aliases"), col("us_aliases"))
        )
        .withColumn(
            "id_number_match", fuzzy_match_udf(col("uk_id_number"), col("us_id_number"))
        )
        .withColumn(
            "nationality_match",
            fuzzy_match_udf(col("uk_nationality"), col("us_nationality")),
        )
        .withColumn(
            "pob_match",
            fuzzy_match_udf(col("uk_place_of_birth"), col("us_place_of_birth")),
        )
        .withColumn("dates_match", dates_udf(col("uk_dob"), col("us_dob")))
        .withColumn("fuzzy_mean_score", mean_cols(array(*cols_req)))
    )
    (
        matches.select(
            "ofac_id",
            "uk_id",
            "fuzzy_mean_score",
            "us_name",
            "uk_name",
            "name_match",
            "us_address",
            "uk_address",
            "address_match",
            "us_countries",
            "uk_countries",
            "countries_match",
            "us_aliases",
            "uk_aliases",
            "aliases_match",
            "us_id_number",
            "uk_id_number",
            "id_number_match",
            "us_nationality",
            "uk_nationality",
            "nationality_match",
            "us_place_of_birth",
            "uk_place_of_birth",
            "pob_match",
            "us_dob",
            "uk_dob",
            "dates_match",
        )
        .filter(col("fuzzy_mean_score") > threshold)
        .write.csv(output)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuzzy Match")
    parser.add_argument("--threshold", "-t", help="fuzzy match threshold", default=50)
    parser.add_argument(
        "--ofac", "-o", help="ofac file location", default="data/ofac.jsonl.gz"
    )
    parser.add_argument(
        "--uk", "-u", help="uk file location", default="data/gbr.jsonl.gz"
    )
    parser.add_argument(
        "--destination", "-d", help="output location", default="data/output.csv"
    )
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    main(args.threshold, args.ofac, args.uk, args.destination)

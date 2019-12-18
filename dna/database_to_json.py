import json
import pymongo
import re
from bson import json_util
from dateutil.parser import parse
import collections
import os


try:
    real_mongo_port = int(os.environ['REAL_MONGO_PORT'])
    lab_hostname = os.environ['LAB_HOSTNAME']
except Exception as E:
    print("ERROR: environment variables not set")
    raise E


def flatten(d, parent_key='', sep='_'):
    """
    This flattens a dictionary
    :param d: the dictionary to be flattened
    :param parent_key: the token used to indicate it came from a previous key
    :param sep: the seperator between parent and child
    :return: a flattened non-string dictionary
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    items = dict(items)
    # remove info like PCA primitive ID
    items_not_strings = {k: v for k, v in items.items() if type(v) != str}
    return dict(items_not_strings)


class DatabaseToJson:
    def __init__(self):
        self.connect_to_mongo()

    def connect_to_mongo(self, host_name=lab_hostname, mongo_port=real_mongo_port):
        """
        Connects and returns a session to the mongo database
        :param host_name: the host computer that has the database server
        :param mongo_port: the port number of the database
        :return: a MongoDB session
        """
        try:
            self.mongo_client = pymongo.MongoClient(host_name, mongo_port)
        except Exception as e:
            print("Cannot connect to the Mongo Client at port {}. Error is {}".format(mongo_port, e))

    def get_primitives_used(self, pipeline_run):
        """
        A helper function for getting the primitives used
        :param pipeline_run: a dictionary-like object containing the pipeline run
        :return: a list of strings where each string is the python path of the primitive
        """
        primitives = []
        for step in pipeline_run['steps']:
            primitives.append(step['primitive']['python_path'])
        return primitives

    def get_time_elapsed(self, pipeline_run):
        """
        A helper function for finding the time of the pipeline_run
        :param pipeline_run: a dictionary-like object containing the pipeline run
        :return: the total time in seconds it took the pipeline to execute
        """
        begin = pipeline_run["steps"][0]["method_calls"][0]["start"]
        begin_val = parse(begin)
        end = pipeline_run["steps"][-1]["method_calls"][-1]["end"]
        end_val = parse(end)
        total_time = (end_val - begin_val).total_seconds()
        return total_time

    def get_pipeline_from_run(self, pipeline_run):
        """
        This function gets the pipeline that corresponds to a pipeline run
        :param pipeline_run: the produce pipeline run
        :return: the pipeline that corresponds to the pipeline_run
        """
        db = self.mongo_client.metalearning
        collection = db.pipelines
        pipeline_doc = collection.find({"$and": [{"id": pipeline_run["pipeline"]["id"]},
                                                 {"digest": pipeline_run["pipeline"]["digest"]}]})[0]
        return pipeline_doc

    def get_pipeline_run_info(self, pipeline_run):
        """
        Collects and gathers the data needed for the DNA system from a pipeline run
        :param pipeline_run: the pipeline run object to be summarized
        :return: a dictionary object summarizing the pipeline run
        """
        pipeline = self.get_pipeline_from_run(pipeline_run)
        pipeline_id = pipeline["id"]
        simple_pipeline  = self.parse_simpler_pipeline(pipeline)
        problem_type = self.get_problem_type_from_pipeline(pipeline)
        raw_dataset_name = pipeline_run["datasets"][0]["id"]
        test_accuracy = pipeline_run["run"]["results"]["scores"][0]["value"]
        test_predict_time = self.get_time_elapsed(pipeline_run)
        train_accuracy = 0
        train_predict_time = 0  # TODO have this find the fit pipeline and get the time
        pipeline_run_info = {
                                "pipeline": simple_pipeline,
                                "pipeline_id": pipeline_id,
                                "problem_type": problem_type,
                                "raw_dataset_name": raw_dataset_name,
                                "test_accuracy": test_accuracy,
                                "test_time": test_predict_time,
                                "train_accuracy": train_accuracy,
                                "train_time": train_predict_time
                            }
        return pipeline_run_info

    def get_metafeature_info(self, pipeline_run):
        """
        Collects and gathers the data needed for the DNA system from a metafeature
        :param dataset_name: the name/id of the dataset
        :return: a dictionary object summarizing the dataset in metafeatures
        """
        db = self.mongo_client.metalearning
        collection = db.metafeatures
        try:
            metafeatures = collection.find({"$and": [{"datasets.id": pipeline_run["datasets"][0]["id"]},
                                                     {"datasets.digest": pipeline_run["datasets"][0]["digest"]}]})[0]
            features = metafeatures["steps"][2]["method_calls"][1]["metadata"]["produce"][0]["metadata"]["data_metafeatures"]
            features_flat = flatten(features)
            # TODO: implement this
            metafeatures_time = 0
            return {"metafeatures": features_flat, "metafeatures_time": metafeatures_time}
        except Exception as e:
            # don't use this pipeline_run
            return {}

    def collect_pipeline_runs(self):
        """
        This is the main function that collects, and writes to file, all pipeline runs and metafeature information
        It writes the file to data/complete_pipelines_and_metafeatures.json
        :param mongo_client: a connection to the Mongo database
        """
        db = self.mongo_client.metalearning
        collection = db.pipeline_runs
        collection_size = collection.count()
        pipeline_cursor = collection.find()
        list_of_experiments = {"classification": [], "regression": []}
        for index, pipeline_run in enumerate(pipeline_cursor):
            if index % 1000 == 0:
                print("At {} out of {} documents".format(index, collection_size))
                # if index == 2000:
                #     # running into memory errors
                #     break
            pipeline_run_info = self.get_pipeline_run_info(pipeline_run)
            metafeatures = self.get_metafeature_info(pipeline_run)
            # TODO: get all metafeatures so we don't need this
            if metafeatures != {}:
                experiment_json = dict(pipeline_run_info, **metafeatures)
                list_of_experiments[experiment_json["problem_type"]].append(experiment_json)

        for problem_type in list_of_experiments.keys():
            final_data_file = json.dumps(list_of_experiments[problem_type], sort_keys=True, indent=4, default=json_util.default)
            with open("data/complete_pipelines_and_metafeatures_test_{}.json".format(problem_type), "w") as file:
                file.write(final_data_file)

        return

    def is_phrase_in(self, phrase, text):
        """
        A simple regex search
        :param phrase: the phrase to search for
        :param text: the text to be searched
        :return:
        """
        return re.search(r"\b{}\b".format(phrase), text, re.IGNORECASE) is not None

    def get_problem_type_from_pipeline(self, pipeline):
        """
        This function finds the problem type from the pipeline steps
        :param pipeline: the full d3m pipeline
        :return: a string containing the type of problem
        """
        is_classification = self.is_phrase_in("d3m.primitives.classification", json.dumps(pipeline['steps']))
        is_regression = self.is_phrase_in("d3m.primitives.regression", json.dumps(pipeline['steps']))
        if is_classification and is_regression:
            print("Cannot be both")
            raise Exception
        elif is_classification:
            predictor_model = "classification"
        elif is_regression:
            predictor_model = "regression"
        else:
            print("Cannot be none")
            raise Exception

        return predictor_model

    def parse_simpler_pipeline(self, full_pipeline):
        """
        This function takes a pipeline object from D3M and turns it into a list of dictionaries where
        each dictionary is a primitive containing the primitive name and the inputs (a list of ints)
        :param full_pipeline: the full d3m pipeline
        :return: The simplified pipeline
        """
        pipeline_steps = full_pipeline["steps"]
        simple_pipeline = []
        for pipeline_step in pipeline_steps:
            pipeline_step_name = pipeline_step["primitive"]["python_path"]
            inputs_list = []
            for key, value in pipeline_step["arguments"].items():
                string_name = value["data"]
                pipeline_step_inputs = self.parse_input_string(string_name)
                inputs_list.append(pipeline_step_inputs)
            # add info to our pipeline
            simple_pipeline.append({"name": pipeline_step_name, "inputs": inputs_list})

        return simple_pipeline

    def parse_input_string(self, string_name):
        """
        This helper function parses the input name from the D3M version (aka `steps.0.produce` to 0)
        :param string_name: the string name from D3M
        :return: the simplified name of the input
        """
        list_of_parts = string_name.split(".")
        if list_of_parts[0] == "inputs":
            return string_name
        else:
            # return only the integer part
            return int(list_of_parts[1])


if __name__ == "__main__":
    db_to_json = DatabaseToJson()
    db_to_json.collect_pipeline_runs()
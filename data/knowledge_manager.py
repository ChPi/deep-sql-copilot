import os

import pandas as pd
from openai import OpenAI
from pymilvus import AnnSearchRequest
from pymilvus import DataType, Function, FunctionType, RRFRanker
from pymilvus import MilvusClient

from data import LlmConfigManager, DatabaseManager

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "milvus.db")
print(db_path)
client = None


def get_milvus_client():
    global client
    if client is None:
        client = MilvusClient(db_path)
    return client


analyzer_params = {"tokenizer": "standard", "filter": ["lowercase", "cnalphanumonly"]}
schema = MilvusClient.create_schema()
embedding_model_id = "embedding"

config = LlmConfigManager()
embedding_config = config.get_embedding_config("embedding")
if embedding_config:
    embedding_model = OpenAI(base_url=embedding_config.base_url, api_key=embedding_config.api_key)
else:
    embedding_model = None
    print("Warning: Embedding configuration not found")


def create_column_collection(database):
    """存储列的信息"""
    client = get_milvus_client()
    collection_name = f"{database}_column"
    schema = MilvusClient.create_schema()
    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )
    schema.add_field(
        field_name="comment",
        datatype=DataType.VARCHAR,
        max_length=65000,
        analyzer_params=analyzer_params,
        enable_match=True,  # Enable text matching
        enable_analyzer=True,  # Enable text analysis
    )
    schema.add_field(field_name="bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,  # Dimension for text-embedding-3-small
    )

    # Define BM25 function to generate sparse vectors from text
    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["comment"],
        output_field_names="bm25",
    )

    # Add the function to schema
    schema.add_function(bm25_function)
    # Define indexes
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")

    if client.has_collection(collection_name):
        return
    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def create_entity_collection(database):
    """存储维度字典信息"""
    client = get_milvus_client()
    collection_name = f"{database}_entity"
    schema = MilvusClient.create_schema()
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    )
    schema.add_field(
        field_name="column_id",
        datatype=DataType.INT64,
    )
    schema.add_field(
        field_name="content",
        datatype=DataType.VARCHAR,
        max_length=100,
        analyzer_params=analyzer_params,
        enable_match=True,  # Enable text matching
        enable_analyzer=True,  # Enable text analysis
    )
    schema.add_field(field_name="bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,
    )
    # Define BM25 function to generate sparse vectors from text
    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["content"],
        output_field_names="bm25",
    )

    # Add the function to schema
    schema.add_function(bm25_function)
    # Define indexes
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")

    if client.has_collection(collection_name):
        return
    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def create_metric_collection(database):
    """存储指标信息"""
    client = get_milvus_client()
    collection_name = f"{database}_metric"
    schema = MilvusClient.create_schema()
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    )
    schema.add_field(
        field_name="name",
        datatype=DataType.VARCHAR,
        max_length=100,
    )
    schema.add_field(
        field_name="comment",
        datatype=DataType.VARCHAR,
        max_length=65000,
        analyzer_params=analyzer_params,
        enable_match=True,  # Enable text matching
        enable_analyzer=True,  # Enable text analysis
    )
    schema.add_field(field_name="bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,  # Dimension for text-embedding-3-small
    )

    # Define BM25 function to generate sparse vectors from text
    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["comment"],
        output_field_names="bm25",
    )

    # Add the function to schema
    schema.add_function(bm25_function)
    # Define indexes
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")

    if client.has_collection(collection_name):
        return
    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def create_query_collection(database):
    """存储查询信息"""
    client = get_milvus_client()
    collection_name = f"{database}_query"
    schema = MilvusClient.create_schema()
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    )
    schema.add_field(
        field_name="query",
        datatype=DataType.VARCHAR,
        max_length=10240,
        analyzer_params=analyzer_params,
        enable_match=True,  # Enable text matching
        enable_analyzer=True,
    )
    schema.add_field(
        field_name="sql",
        datatype=DataType.VARCHAR,
        max_length=20480,
        analyzer_params=analyzer_params,
        enable_match=True,  # Enable text matching
        enable_analyzer=True,  # Enable text analysis
    )
    schema.add_field(field_name="bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(
        field_name="embedding",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024,  # Dimension for text-embedding-3-small
    )
    # Define BM25 function to generate sparse vectors from text
    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["query"],
        output_field_names="bm25",
    )

    # Add the function to schema
    schema.add_function(bm25_function)
    # Define indexes
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="IP")

    if client.has_collection(collection_name):
        return
    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def create_sql_collection(database):
    """存储查询信息"""
    client = get_milvus_client()
    collection_name = f"{database}_sql"
    schema = MilvusClient.create_schema()
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    )
    schema.add_field(
        field_name="sql",
        datatype=DataType.VARCHAR,
        max_length=20480,
        analyzer_params=analyzer_params,
        enable_match=True,  # Enable text matching
        enable_analyzer=True,
    )

    schema.add_field(field_name="bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # Define BM25 function to generate sparse vectors from text
    bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["sql"],
        output_field_names="bm25",
    )

    # Add the function to schema
    schema.add_function(bm25_function)
    # Define indexes
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )

    # Drop collection if exist
    if client.has_collection(collection_name):
        return
    # Create the collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


class KnowledgeManager:
    def __init__(self, database):
        self.database_id = database
        self.llm_manager: LlmConfigManager = LlmConfigManager()
        self.database_manager = DatabaseManager()
        self.init()

    def init(self):
        create_query_collection(self.database_id)
        create_metric_collection(self.database_id)
        create_column_collection(self.database_id)
        create_entity_collection(self.database_id)

    def search(self, text, collection_name):
        if not isinstance(text, str):
            return []
        client = get_milvus_client()
        collection_name = f"{self.database_id}{collection_name}"
        vector = self.llm_manager.get_embedding(embedding_model_id, text)
        search_param_1 = {
            "data": [vector],
            "anns_field": "embedding",
            "param": {"nprobe": 10},
            "limit": 10
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": [text],
            "anns_field": "bm25",
            "param": {"drop_ratio_search": 0.2},
            "limit": 10
        }
        request_2 = AnnSearchRequest(**search_param_2)
        reqs = [request_1, request_2]
        # ranker = Function(
        #     name="rrf",
        #     input_field_names=[],
        #     function_type=FunctionType.RERANK,
        #     params={
        #         "reranker": "rrf",
        #         "k": 100
        #     }
        # )
        ranker = RRFRanker(100)

        try:
            res = client.hybrid_search(
                collection_name=collection_name,
                reqs=reqs,
                ranker=ranker,
                limit=10
            )
            return res
        except Exception as e:
            # print(e)
            return []

    def add_query(self, query, sql):
        # query content
        client = get_milvus_client()
        embedding = self.llm_manager.get_embedding(embedding_model_id, query)
        data = [{"query": query, "sql": sql, "embedding": embedding}]
        client.insert(
            collection_name=f"{self.database_id}_query",
            data=data
        )

    def search_query(self, query):
        """搜索问题对，[(query, sql)]"""
        return self.search(query, "_query")

    def search_entity(self, text):
        """通过关键字搜索相关值的字段，返回list,[(col id, value, similar)]"""
        # id: 1, distance: 0.006047376897186041, entity
        return self.search(text, "_entity")

    def add_column(self, list):
        data = []
        client = get_milvus_client()
        for i in list:
            column_id = i["id"]
            comment = i["column_comment"]
            if not comment:
                comment = "null"
            embedding = self.llm_manager.get_embedding(embedding_model_id, comment)
            data.append({"id": column_id, "comment": comment, "embedding": embedding})

        client.insert(
            collection_name=f"{self.database_id}_column",
            data=data
        )

    def search_and_embedding_col(self, text):
        """通过关键字搜索相关字段, 如果为varchar类型则把数据库的数据存储到向量数据
            返回list，[(col id, comment)]"""
        search_data = self.search(text, "_column")
        column_id_list = [i["id"] for i in search_data[0]]
        # id, database_id, table_name, column_name,
        #         data_type, column_comment,
        col_data = self.database_manager.get_by_column_id(column_id_list).to_dict('records')
        engine = self.database_manager.get_engine(self.database_id)
        client = get_milvus_client()

        for col in col_data:
            if col["data_type"] not in ("varchar", "string", "text"):
                continue
            id = col["id"]
            table_name = col["table_name"]
            column_name = col["column_name"]
            if col["embedding"]:
                continue
            sql = f"""
            select {column_name} as entity from {self.database_id}.{table_name} group by {column_name} order by count(*) desc limit 100
            """
            l = pd.read_sql(sql, engine)["entity"].tolist()
            insert_data = []
            for e in l:
                if not e:
                    continue
                embedding = self.llm_manager.get_embedding(embedding_model_id, e)
                insert_data.append({"column_id": id, "content": e, "embedding": embedding})
            client.insert(
                collection_name=f"{self.database_id}_entity",
                data=insert_data
            )
            self.database_manager.set_embedding(id)
        return col_data

    def add_sql(self, sql_list):
        client = get_milvus_client()
        data = []
        for sql in sql_list:
            data.append({"sql": sql})
        client.insert(
            collection_name=f"{self.database_id}_sql",
            data=data
        )
        return

    def search_sql(self, keywords):
        """根据关键字搜索历史sql"""
        client = get_milvus_client()
        search_params = {
            'params': {'drop_ratio_search': 0.2},
        }
        if not keywords:
            return []
        res = client.search(
            collection_name=f"{self.database_id}_sql",
            data=" ".join(keywords),
            anns_field='bm25',
            output_fields=['sql'],  # Fields to return in search results; sparse field cannot be output
            limit=5,
            search_params=search_params
        )
        return res

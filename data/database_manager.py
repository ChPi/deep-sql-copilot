import json
import logging
import os
import urllib.parse
from typing import Dict, List, Any

import pandas as pd
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    id: str
    name: str
    type: str  # sqlite, mysql
    host: str
    port: int
    username: str
    password: str
    database: str


class TableSchema(BaseModel):
    table_name: str
    columns: List[Dict[str, Any]]
    table_comment: str


class Column(BaseModel):
    id: str
    table_name: str
    comment: str
    data_type: str


class DatabaseManager:

    def __init__(self, config_file: str = "config/database_config.json"):
        self.config_file = config_file
        self.databases: Dict[str, DatabaseConfig] = {}
        self.connections: Dict[str, Any] = {}
        self.schemas: Dict[str, Dict[str, TableSchema]] = {}
        self.load_config()

    def get_schema(self, database, table):
        """获取制定表的schema"""
        # todo
        if database not in self.schemas:
            self.get_table_schemas(database)

        schema = self.schemas[database]
        return schema.get(table)

    def load_config(self):
        """加载数据库配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    for config in configs:
                        db_config = DatabaseConfig(**config)
                        self.databases[db_config.id] = db_config
            else:
                # 创建默认配置目录
                os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
                self.save_config()
        except Exception as e:
            logger.error(f"加载数据库配置失败: {e}")

    def get_engine(self, database_id: str):
        """获取数据库连接"""
        try:
            if database_id in self.connections:
                return self.connections[database_id]

            if database_id not in self.databases:
                raise ValueError(f"数据库不存在: {database_id}")

            db_config = self.databases[database_id]

            if db_config.type == "sqlite":
                conn = create_engine(f"sqlite:///{db_config.database}")
                self.connections[database_id] = conn

                return conn
            elif db_config.type == "mysql":
                password = urllib.parse.quote_plus(db_config.password)
                conn = create_engine(
                    f"mysql+pymysql://{db_config.username}:{password}"
                    f"@{db_config.host}:{db_config.port}/{db_config.database}"
                    f"?charset=utf8mb4"
                )
                self.connections[database_id] = conn
                return conn

        except Exception as e:
            logger.error(f"获取数据库连接失败: {e}")
            raise

    def get_table_schemas(self, database_id: str) -> Dict[str, TableSchema]:
        """获取数据库的所有表结构信息"""
        if database_id in self.schemas:
            return self.schemas[database_id]
        try:
            engine = self.get_engine(database_id)

            # 获取所有表名
            if self.databases[database_id].type == "mysql":
                table_query = "SHOW TABLES"
            elif self.databases[database_id].type == "sqlite":
                table_query = "SELECT name FROM sqlite_master WHERE type='table'"
            else:
                raise ValueError(f"不支持的数据库类型: {self.databases[database_id].type}")

            tables = pd.read_sql(table_query, engine)
            table_column = tables.columns[0]
            schemas = {}

            # 获取每个表的列信息
            for table_name in tables[table_column]:
                table_comment = ""

                # 获取表注释
                if self.databases[database_id].type == "mysql":
                    # 获取表注释
                    table_comment_query = f"""
                    SELECT TABLE_COMMENT
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = '{self.databases[database_id].database}'
                    AND TABLE_NAME = '{table_name}'
                    """
                    table_comment_result = pd.read_sql(table_comment_query, engine)
                    if not table_comment_result.empty:
                        table_comment = table_comment_result.iloc[0]['TABLE_COMMENT'] or ""

                if self.databases[database_id].type == "mysql":
                    column_query = f"""
                    SELECT
                        COLUMN_NAME as column_name,
                        DATA_TYPE as data_type,
                        COLUMN_COMMENT as column_comment,
                        IS_NULLABLE as is_nullable,
                        COLUMN_KEY as column_key,
                        COLUMN_DEFAULT as column_default
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = '{self.databases[database_id].database}'
                    AND TABLE_NAME = '{table_name}'
                    ORDER BY ORDINAL_POSITION
                    """
                elif self.databases[database_id].type == "sqlite":
                    column_query = f"PRAGMA table_info('{table_name}')"

                columns_df = pd.read_sql(column_query, engine)
                columns = []

                for _, row in columns_df.iterrows():
                    if self.databases[database_id].type == "mysql":
                        # 将表注释拼接到列注释中
                        column_comment = row["column_comment"] or ""
                        if table_comment and column_comment:
                            combined_comment = f"{column_comment} [表注释: {table_comment}]"
                        elif table_comment:
                            combined_comment = f"[表注释: {table_comment}]"
                        else:
                            combined_comment = column_comment

                        column_info = {
                            "column_name": row["column_name"],
                            "data_type": row["data_type"],
                            "comment": combined_comment,
                            "is_nullable": row["is_nullable"],
                            "is_primary": "PRI" in row["column_key"],
                            "default_value": row["column_default"]
                        }
                    elif self.databases[database_id].type == "sqlite":
                        column_info = {
                            "column_name": row["name"],
                            "data_type": row["type"],
                            "comment": "",
                            "is_nullable": row["notnull"] == 0,
                            "is_primary": row["pk"] == 1,
                            "default_value": row["dflt_value"]
                        }
                    columns.append(column_info)

                schemas[table_name] = TableSchema(table_name=table_name, columns=columns, table_comment=table_comment)
            self.schemas[database_id] = schemas
            return schemas

        except Exception as e:
            logger.error(f"获取表结构失败: {e}")
            raise

    def create_system_tables(self):
        """创建system数据库中的表结构存储表"""
        try:
            system_engine = self.get_engine("system")

            # 创建表结构存储表
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS table_schemas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                database_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                column_name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                column_comment TEXT,
                is_nullable TEXT,
                is_primary BOOLEAN,
                default_value TEXT,
                embedding BOOLEAN DEFAULT false,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(database_id, table_name, column_name)
            )
            """
            with system_engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()

        except Exception as e:
            logger.error(f"创建system表失败: {e}")
            raise

    def save_table_schemas_to_system(self, database_id: str):
        """将数据库的表结构信息保存到system数据库"""
        try:
            # 获取目标数据库的表结构
            schemas = self.get_table_schemas(database_id)

            # 获取system数据库连接
            system_engine = self.get_engine("system")

            # 确保system表存在
            self.create_system_tables()

            # 清空该数据库的旧记录
            delete_sql = text("DELETE FROM table_schemas WHERE database_id = :id")
            with system_engine.connect() as conn:
                conn.execute(delete_sql, {"id": database_id})
                conn.commit()

            # 插入新的表结构信息
            for table_name, table_schema in schemas.items():
                for column in table_schema.columns:
                    insert_sql = text("""
                    INSERT INTO table_schemas
                    (database_id, table_name, column_name, data_type, column_comment, is_nullable, is_primary, default_value)
                    VALUES (:database_id, :table_name, :column_name, :data_type, 
                    :comment, :is_nullable, :is_primary, :default_value)
                    """)
                    with system_engine.connect() as conn:
                        conn.execute(insert_sql, {
                            "database_id": database_id,
                            "table_name": table_name,
                            "column_name": column["column_name"],
                            "data_type": column["data_type"],
                            "comment": column["comment"],
                            "is_nullable": column["is_nullable"],
                            "is_primary": column["is_primary"],
                            "default_value": column["default_value"]
                        })
                        conn.commit()

            logger.info(f"成功保存数据库 {database_id} 的表结构信息到system数据库")

        except Exception as e:
            logger.error(f"保存表结构到system数据库失败: {e}")
            raise

    def set_embedding(self, column_id):
        system_engine = self.get_engine("system")
        update_sql = text("""
                    update table_schemas set embedding=true where id=:id
                    """)
        with system_engine.connect() as conn:
            conn.execute(update_sql, {
                "id": column_id
            })
            conn.commit()

    def get_column(self, database_id):
        sql = f"""
        select id, database_id, table_name, column_name, 
        data_type, column_comment, is_nullable, is_primary, default_value
        from table_schemas where database_id = '{database_id}'
        """
        system_engine = self.get_engine("system")
        df = pd.read_sql(sql, system_engine)
        return df

    def get_by_column_id(self, column_id_list):
        sql = f"""
        select id, database_id, table_name, column_name, 
        data_type, column_comment, is_nullable, is_primary, default_value, embedding
        from table_schemas where id in ({",".join(str(i) for i in column_id_list)})
        order by database_id,table_name,id
        """
        system_engine = self.get_engine("system")
        df = pd.read_sql(sql, system_engine)
        return df

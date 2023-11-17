import singlestoredb as s2
from sqlalchemy import create_engine

def get_sql_alchemy_connection(host: str, port: int, database: str, user: str, password: str,
                               dialect='mysql+pymysql') -> object:
    """
    Function to connect to SingleStore.Can be used with Pandas, but does not work well if you have vector data.
    For vector data use `get_connection()` method
    Args:
        host: S2 Host URL
        port: S2 PORT
        database: S2 DB name
        user: S2 usename
        password: S2 Password
        dialect: Dialect to construct the protocol string

    Returns:
        SqlAlchemy Connection
    """

    connection_uri = f"{dialect}://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(connection_uri)
    
    return engine


def get_connection(host: str, port: int, database: str, user: str, password: str, autocommit: bool = True) -> object:
    """
    Function to connect to s2 using native S2 connection. Works well for indexing vectors.
    Args:
        host: S2 Host URL
        port: S2 PORT
        database: S2 DB name
        user: S2 usename
        password: S2 Password
        autocommit: Autocommit on each insert

    Returns:

    """
    return s2.connect(host=host, user=user, port=port, password=password, database=database, autocommit=autocommit)


def get_cursor(host: str, port: int, database: str, user: str, password: str) -> object:
    connection = get_connection(host=host, user=user, port=port, password=password, database=database)
    return connection.cursor()

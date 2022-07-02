from ariadne import MutationType, QueryType

MAIN_TYPEDEF = """
    type Query {
        models: [Model]!
        models_by_version(version: String): [Model]!
        model(id: ID!): Model!
        authors: [Author]!
        author(id: ID!): Author!
        categories: [Category]!
        category(id: ID!): Category!
    }
    """

query = QueryType()
mutation = MutationType()

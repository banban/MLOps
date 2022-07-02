from typing import TypedDict

class Model(TypedDict):
    id: int
    title: str
    content: str
    author_id: int
    category_id: int
    version: str
    parameters: str


class ModelPayload(TypedDict, total=False):
    title: str
    content: str
    version: str

class Author(TypedDict):
    id: int
    name: str

class Category(TypedDict):
    id: int
    name: str

MODELS: dict[int, Model] = {
    1: {"id": 1, "title": "Model 1.3", "author_id": 1, "category_id": 1, "content": "Content 1", "version": "3"},
    2: {"id": 2, "title": "Model 2.3", "author_id": 2, "category_id": 1, "content": "Content 2", "version": "3"},
    3: {"id": 3, "title": "Model 1.4", "author_id": 3, "category_id": 3, "content": "Content 3", "version": "4"},
    4: {"id": 4, "title": "Model 2.4", "author_id": 3, "category_id": 2, "content": "Content 4", "version": "4"},
}

AUTHORS: dict[int, Author] = {
    1: {"id": 1, "name": "Author 1"},
    2: {"id": 2, "name": "Author 2"},
    3: {"id": 3, "name": "Zhibin Liao​"},
}

CATEGORIES: dict[int, Category] = {
    1: {"id": 1, "name": "Deep Learning"},
    2: {"id": 2, "name": "XGBoost"},
    3: {"id": 3, "name": "Tensor Flow"},
}

class NotFoundError(Exception):
    pass


def all_models() -> list[Model]:
    return list(MODELS.values())

def all_models_by_version(version:str) -> Model:

    filtered = filter(lambda x: x['version'] == version, list(MODELS.values()))  #filter(lambda x: x['version'] == version, MODELS)
    #my_item = next((item for item in MODELS if item['version'] == version), None)
    #print(list(filtered).count())
    #if list(filtered).count()==0:
    #    raise NotFoundError("Models by version {version} not found")
    return list(filtered)


def get_model(model_id: int) -> Model:
    if not MODELS.get(model_id):
        raise NotFoundError("Model not found")
    return MODELS[model_id]


def update_model(model_id: int, payload: ModelPayload) -> Model:
    model = MODELS.get(model_id)
    if not model:
        raise NotFoundError("Model not found")
    for key, value in payload.items():
        model[key] = value
    return model


def all_authors() -> list[Author]:
    return list(AUTHORS.values())


def get_author(author_id: int) -> Author:
    if not AUTHORS.get(author_id):
        raise NotFoundError("Author not found")
    return AUTHORS[author_id]


def all_categories() -> list[Category]:
    return list(CATEGORIES.values())


def get_category(category_id: int) -> Category:
    if not CATEGORIES.get(category_id):
        raise NotFoundError("Category not found")
    return CATEGORIES[category_id]

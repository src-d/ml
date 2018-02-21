def readonly(_dict: dict):
    return frozenset(_dict.items())

dataset = [
    {"d": "1", "t": "1", "v": 3},
    {"d": "1", "t": "2", "v": 2},
    {"d": "1", "t": "3", "v": 1},
    {"d": "2", "t": "1", "v": 4},
    {"d": "2", "t": "2", "v": 5},
    {"d": "3", "t": "1", "v": 6},
    {"d": "4", "t": "1", "v": 4},
    {"d": "4", "t": "2", "v": 3},
    {"d": "4", "t": "3", "v": 2},
    {"d": "4", "t": "4", "v": 1},
] * 10

term_freq_result = {
    readonly({"d": "1", "t": "1", "v": 30}),
    readonly({"d": "1", "t": "2", "v": 20}),
    readonly({"d": "1", "t": "3", "v": 10}),
    readonly({"d": "2", "t": "1", "v": 40}),
    readonly({"d": "2", "t": "2", "v": 50}),
    readonly({"d": "3", "t": "1", "v": 60}),
    readonly({"d": "4", "t": "1", "v": 40}),
    readonly({"d": "4", "t": "2", "v": 30}),
    readonly({"d": "4", "t": "3", "v": 20}),
    readonly({"d": "4", "t": "4", "v": 10}),
}

doc_freq_result = {
    "1": 4,
    "2": 3,
    "3": 2,
    "4": 1,
}

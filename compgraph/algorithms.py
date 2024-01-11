from . import Graph, operations


def word_count_graph(name: str, text_column: str = 'text', count_column: str = 'count', file: bool = False) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return Graph.make_graph(name, file) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf', file: bool = False) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    split_word = Graph.make_graph(name, file) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))
    count_docs = Graph.make_graph(name, file) \
        .reduce(operations.Count('docs_count'), [])
    count_idf = split_word.sort([doc_column, text_column]) \
        .reduce(operations.FirstReducer(), [doc_column, text_column]) \
        .sort([text_column]) \
        .reduce(operations.Count('docs_with_word'), [text_column]) \
        .join(operations.InnerJoiner(), count_docs, []) \
        .map(operations.IDF(['docs_count', 'docs_with_word'], 'idf'))
    tf = split_word.sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column), [doc_column])
    return tf.sort([text_column]) \
        .join(operations.InnerJoiner(), count_idf, [text_column]) \
        .map(operations.Product(['tf', 'idf'], result_column)) \
        .sort([text_column]) \
        .reduce(operations.TopN(result_column, n=3), [text_column]) \
        .map(operations.Project([doc_column, text_column, result_column]))


def pmi_graph(name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi', file: bool = False) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    split_word = Graph.make_graph(name, file) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .map(operations.Filter(condition=lambda row: len(row[text_column]) > 4)) \
        .sort([doc_column, text_column]) \
        .reduce(operations.Count('word_in_doc_count'), [doc_column, text_column]) \
        .map(operations.Filter(condition=lambda row: row['word_in_doc_count'] >= 2)) \
        .map(operations.Reveal('word_in_doc_count'))

    freq_of_word_in_doc = split_word.sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column), [doc_column])

    freq_of_word_in_all = split_word.reduce(operations.TermFrequency(text_column, 'freq_in_all'), []) \
        .map(operations.Project([text_column, 'freq_in_all']))

    merged = freq_of_word_in_doc.sort([text_column]) \
        .join(operations.InnerJoiner(), freq_of_word_in_all.sort([text_column]), [text_column]) \
        .map(operations.PMI(['tf', 'freq_in_all'], result_column))
    return merged.sort([doc_column]) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .sort([doc_column]) \
        .reduce(operations.TopN(result_column, 10), [doc_column]) \
        .map(operations.Inverse(result_column)) \
        .sort([doc_column, result_column]) \
        .map(operations.Inverse(result_column))


def yandex_maps_graph(name1: str, name2: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed', file: bool = False) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    time = Graph.make_graph(name1, file) \
        .map(operations.GetDuration(enter_time_column, leave_time_column, 'duration')) \
        .map(operations.GetWeekdayAndHour(enter_time_column, weekday_result_column, hour_result_column)) \
        .map(operations.Project([edge_id_column, 'duration', weekday_result_column, hour_result_column]))

    length = Graph.make_graph(name2, file) \
        .map(operations.GetHaversineDist(start_coord_column, end_coord_column, 'distance')) \
        .map(operations.Project([edge_id_column, 'distance']))

    merge = time.sort([edge_id_column]) \
        .join(operations.InnerJoiner(), length.sort([edge_id_column]), [edge_id_column]) \
        .sort([weekday_result_column, hour_result_column])

    total_dist = merge.reduce(operations.Sum('distance'), [weekday_result_column, hour_result_column])
    total_duration = merge.reduce(operations.Sum('duration'), [weekday_result_column, hour_result_column])

    return total_dist.sort([weekday_result_column, hour_result_column]) \
        .join(operations.InnerJoiner(), total_duration.sort([weekday_result_column,
                                                             hour_result_column]), ['weekday', 'hour']) \
        .map(operations.GetAverageSpeed('distance', 'duration', speed_result_column)) \
        .map(operations.Project([weekday_result_column, hour_result_column, speed_result_column]))

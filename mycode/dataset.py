from typing import NamedTuple, List, Tuple

from mycode.multi_ctgan import MetaBuilder, MultiTableCTGAN


class DatabaseTest:
    path_1k = './mycode/testcode/1k.db'
    path_10k = './mycode/testcode/10k.db'
    path_100k = './mycode/testcode/100k.db'


class Database:
    path_10k = './mycode/data_sqlite.db'
    path_100k = './mycode/100k_data_sqlite.db'
    path_1k = './mycode/1k_data_sqlite.db'


x_args_type = NamedTuple('x_args_type', [
    ('x_table', List[str]),
    ('x_key', List[str]),
    ('x_how', List[str]),
    ('meta_datetime_escapes', List[Tuple[str, str]]),
    ('meta_time_escapes', List[Tuple[str, str]])
])


class XArgs:
    tables_6 = x_args_type(
        x_table=["BookLoan", "Book", "Library", "Student", "Enrollment", "Submission"],
        x_key=['book_id', "library_id", "student_id", "student_id", "student_id"],
        x_how=['inner' for _ in range(5)],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[],
    )
    tables_12 = x_args_type(
        x_table=["BookLoan",
                 "Book", "Library", "Student",
                 "Enrollment", "Submission", "Course",
                 # "Assignment"
                 "CourseTextbook", "Textbook",
                 "Schedule", "Professor"],
        x_key=[
            'book_id', "library_id", "student_id",
            "student_id", "student_id", "course_id",
            # "assignment_id",
            "course_id", "textbook_id",
            "course_id", "professor_id"],
        x_how=['inner' for _ in range(10)],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[("Schedule", "time_slot")]
    )

    tables_14 = x_args_type(
        x_table=["BookLoan",
                 "Book", "Library", "Student",
                 "Enrollment", "Submission", "Course",
                 # "Assignment"
                 "CourseTextbook", "Textbook",
                 "Schedule", "Professor", 'ProjectMember',
                 'ResearchProject', 'ResearchGroup'],
        x_key=[
            'book_id', "library_id", "student_id",
            "student_id", "student_id", "course_id",
            # "assignment_id",
            "course_id", "textbook_id",
            "course_id", "professor_id", "professor_id",
            "project_id", "group_id"],
        x_how=['inner' for _ in range(13)],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[("Schedule", "time_slot")]
    )


class XMetaBuilder(MetaBuilder):
    def __init__(self, x_args: x_args_type):
        super().__init__()
        self.x_args = x_args

    def build(self, multi_wrapper, metadata):
        x_args = self.x_args

        def remove_copy_tag(key: str):
            index = key.find("_COPY")
            if index != -1:
                return key[:index]
            else:
                return key

        # datetime key
        escapes_columns = list(x_args.meta_datetime_escapes.copy())
        escapes_columns.extend(x_args.meta_time_escapes)
        escape_key = [x[0] + MultiTableCTGAN.SEPERATOR + x[1] for x in set(escapes_columns)]
        metadata.datetime_format = {
            key: "%Y-%m-%d" for key in metadata.datetime_columns if remove_copy_tag(key) not in escape_key
        }

        copyed_columns = {
            'datetime': [key for key in metadata.datetime_columns if
                         tuple(remove_copy_tag(key).split("_TABLE_")) in x_args.meta_datetime_escapes],
            'time': [key for key in metadata.datetime_columns if
                     tuple(remove_copy_tag(key).split("_TABLE_")) in x_args.meta_time_escapes],
        }
        print(copyed_columns)

        metadata.datetime_format.update({
            i: "%Y-%m-%d %H:%M:%S" for i in copyed_columns['datetime']
        })
        # metadata.datetime_format.update({
        #     i: '' for i in copyed_columns['time']  # #"%H:%M:%S"
        #     # 此处有问题，SDG不支持该格式
        # })
        metadata.discrete_columns = set([
            key for key in metadata.discrete_columns if key not in metadata.datetime_columns
        ])
        print(
            f"{metadata.int_columns=}\n{metadata.float_columns=}\n{metadata.const_columns=}\n{metadata.bool_columns=}\n{metadata.discrete_columns=}")

        metadata.column_encoder = {
            key: "label" for key in metadata.discrete_columns
        }
        return metadata

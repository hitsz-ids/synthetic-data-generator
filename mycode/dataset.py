from typing import NamedTuple, List, Tuple

from multi_ctgan import MetaBuilder, MultiTableCTGAN


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

        # datetime key
        escapes_columns = x_args.meta_datetime_escapes
        escapes_columns.extend(x_args.meta_time_escapes)
        escape_key = [x[0] + MultiTableCTGAN.SEPERATOR + x[1] for x in escapes_columns]
        metadata.datetime_format = {
            key: "%Y-%m-%d" for key in metadata.datetime_columns if key not in escape_key
        }
        metadata.datetime_format.update({
            i: "%Y-%m-%d %H:%M:%S" for i in x_args.meta_datetime_escapes
        })
        metadata.datetime_format.update({
            i: "%H:%M:%S" for i in x_args.meta_time_escapes
        })
        metadata.discrete_columns = set([
            key for key in metadata.discrete_columns if key not in metadata.datetime_columns
        ])
        print(
            f"{metadata.int_columns=}\n{metadata.float_columns=}\n{metadata.const_columns=}\n{metadata.bool_columns=}\n{metadata.discrete_columns=}")

        metadata.column_encoder = {
            key: "label" for key in metadata.discrete_columns
        }
        return metadata

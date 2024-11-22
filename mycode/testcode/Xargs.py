from typing import List, Tuple

from pydantic.dataclasses import dataclass

from mycode.testcode.metabuilder import MetaBuilder


@dataclass
class XArg:
    x_table: List[str]
    x_key: List[str]
    x_how: List[str]
    meta_id_escapes: List[str]
    meta_datetime_escapes: List[Tuple[str, str]]
    meta_time_escapes: List[Tuple[str, str]]

    def copy(self):
        return XArg(
            self.x_table, self.x_key, self.x_how,
            self.meta_id_escapes, self.meta_datetime_escapes, self.meta_time_escapes)


class XArgs:
    tables_6 = XArg(
        x_table=["BookLoan", "Book", "Library", "Student", "Enrollment", "Submission"],
        x_key=['book_id', "library_id", "student_id", "student_id", "student_id"],
        x_how=['inner' for _ in range(5)],
        meta_id_escapes=["assignment_id", "course_id"],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[],
    )
    tables_11 = XArg(
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
        meta_id_escapes=["assignment_id"],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[("Schedule", "time_slot")]
    )

    tables_14 = XArg(
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
        meta_id_escapes=["assignment_id"],
        meta_datetime_escapes=[("Submission", "submission_date")],
        meta_time_escapes=[("Schedule", "time_slot")]
    )


class XMetaBuilder(MetaBuilder):
    def __init__(self, x_args: XArg):
        super().__init__()
        self.x_args = x_args

    def build(self, multi_wrapper, metadata):
        x_args = self.x_args

        def remove_copy_tag(key: str):
            index = key.find("_COPY")  # TODO 更换为常量COPY_NAME_SEPERATOR
            if index != -1:
                return key[:index]
            else:
                return key

        # datetime key
        escapes_columns = list(x_args.meta_datetime_escapes.copy())
        escapes_columns.extend(x_args.meta_time_escapes)
        escape_key = [MultiTableCTGAN.column_name_encode(x[0], x[1]) for x in set(escapes_columns)]
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

import os

# 多线程相关，目前不用
CPU_COUNT = os.cpu_count()  # 后续用于多线程
PORT = 7970  # 端口，目前未启用
SDG_PORT = PORT

# 日志路径
logger_config_path = "utilities/logging.conf"


# 路径配置，目前不太用
tmp_dir = "/tmp/sdg/tmp/"
sdg_data_dir = "/tmp/sdg/data/"
sdg_log_dir = "/tmp/sdg/log/"
# 分任务进行，目前不用
# sdg_job_dir   = "tmp/job/"

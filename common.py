import os
import shutil
import time


class CommonCls:

    def __init__(self, log_dir=None, param_path=None, delete_previous_log=False):
        current_work_dir = os.getcwd()
        if delete_previous_log:
            self.delete_previous_logs(current_work_dir)
        current_timestamp = time.strftime("%H-%M-%S-%Y-%m-%d", time.localtime())
        self.log_dir = log_dir or os.path.join(current_work_dir, current_timestamp + '-log')
        os.makedirs(self.log_dir, exist_ok=True)
        self.params_file_path = param_path or os.path.join(current_work_dir, 'params.txt')
        self.params = {}
        self.get_parameter()

    def get_parameter(self, param_file_path=None):
        param_file_path = param_file_path or self.params_file_path
        with open(param_file_path, encoding='utf-8') as param_data:
            lines = param_data.readlines()
            for line in lines:
                if line.startswith('#'):
                    pass
                else:
                    line_list = line.strip().split('=')
                    assert len(line_list) == 2
                    param_key, param_value = line_list
                    self.params[param_key] = param_value
        # 将每次运行的配置文件也放进日志目录内
        shutil.copy2(self.params_file_path, os.path.join(self.log_dir, 'params.log'))

    def log(self, text=None, log_name='default', print_log=False):
        log_path = os.path.join(self.log_dir, f'{log_name}.log')
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(log_path, 'a', encoding='utf-8') as output:
            text = f'[{timestamp}]\n{text}\n'
            if print_log:
                print(text)
            output.writelines(text)

    def delete_previous_logs(self, directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path) and item.endswith("log"):
                # 删除子目录及其文件
                self.delete_previous_logs(item_path)
                # 删除空目录
                os.rmdir(item_path)
            elif os.path.isfile(item_path) and item.endswith("log"):
                # 删除文件
                os.remove(item_path)

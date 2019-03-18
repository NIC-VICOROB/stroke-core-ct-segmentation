import os
import subprocess
from tkinter import *
from tkinter.ttk import *
import json

from tkinter import font as tkFont


class OtherGUI:
    def __init__(self):
        self.storage_path = 'storage/'
        self.pad_y = 3
        self.train_errors = {}
        self.test_errors = {}

        # Check if there are pre-trained models
        with open(self.storage_path + 'models.txt', 'r') as models_file:
            models_dict = json.loads(models_file.read())  # use `json.dumps` to do the reverse
            self.models_list = [k for k in models_dict]

        self.window = Tk()
        self.window.title("Stroke lesion core segmentation")
        self.window.geometry('1200x380')

        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=14)
        # self.window.geometry('900x800')
        self.window.option_add("*Font", default_font)

        self.notebook = Notebook(self.window)
        self.notebook.pack(side=TOP, fill=X)

        self.train_tab = Panedwindow(self.notebook)  # first page, which would get widgets gridded into it
        self.train_tab.pack(side=RIGHT, fill=X)

        self.test_tab = Panedwindow(self.notebook)  # second page
        self.test_tab.pack(side=LEFT, fill=X)

        self.notebook.add(self.train_tab, text=' ' * 5 + 'Training' + ' ' * 5)
        self.notebook.add(self.test_tab, text=' ' * 5 + 'Inference' + ' ' * 5)

        self.make_train_tab()
        self.make_test_tab()

        self.window.mainloop()

    def make_test_tab(self):
        # Data frame
        self.test_data_frame = Labelframe(self.test_tab, text='Data')
        self.test_data_frame.pack()

        self.test_datapath_lbl = Label(self.test_data_frame, text="Dataset file: ")
        self.test_datapath_lbl.pack(side=LEFT)

        self.test_datapath_entry = Entry(self.test_data_frame, width=60)
        self.test_datapath_entry.pack(side=LEFT)
        self.test_datapath_entry.config(state='disabled')

        self.test_datapath_browse = Button(self.test_data_frame, text="Browse...", command=self.test_datapath_dialog)
        self.test_datapath_browse.pack(side=LEFT)

        self.test_tab.add(self.test_data_frame)

        # Pre-processing
        self.test_preproc_frame = Labelframe(self.test_tab, text='Pre-processing')
        self.test_preproc_frame.pack()

        self.test_ss_intvar = IntVar()
        self.test_ss_chkbtn = Checkbutton(self.test_preproc_frame, text="Skull stripping (only ISLES18)",
                                          variable=self.test_ss_intvar)
        self.test_ss_chkbtn.grid(row=0, column=0, columnspan=2, sticky='w')

        self.test_tab.add(self.test_preproc_frame)

        # Pretrained_model
        self.test_model_frame = Labelframe(self.test_tab, text='CNN Model')
        self.test_model_frame.pack()

        self.test_pretrain_lbl = Label(self.test_model_frame, text="Pre-trained model: ")
        self.test_pretrain_lbl.grid(column=0, row=0, columnspan=1, pady=self.pad_y)

        self.test_model_stringvar = StringVar()
        self.test_pretrain_combo = Combobox(self.test_model_frame, takefocus=False,
                                            textvariable=self.test_model_stringvar,
                                            postcommand=self.validate_update_pretrained_model_list)
        self.test_pretrain_combo['values'] = ['None', ] + [model_key for model_key in self.models_list]
        self.test_pretrain_combo.current(0)
        self.test_pretrain_combo.configure(state='readonly')
        self.test_pretrain_combo.grid(column=1, row=0, pady=self.pad_y)

        self.test_tab.add(self.test_model_frame)

        # Errors and start button
        self.test_button_frame = Frame(self.test_tab)
        self.test_button_frame.pack()

        self.test_error_stringvar = StringVar()
        self.test_error_lbl = Label(self.test_button_frame, textvariable=self.test_error_stringvar, foreground="#f44")
        self.test_error_lbl.pack(side=LEFT)

        self.test_btn = Button(self.test_button_frame, text="Start inference", command=self.launch_inference)
        self.test_btn.pack(side=RIGHT)

        self.test_tab.add(self.test_button_frame)

    def make_train_tab(self):
        # Data frame
        self.data_frame = Labelframe(self.train_tab, text='Data')
        self.data_frame.pack()

        self.datapath_lbl = Label(self.data_frame, text="Dataset file: ")
        self.datapath_lbl.pack(side=LEFT)

        self.datapath_entry = Entry(self.data_frame, width=60, text='Browse for dataset specification file (.txt)')
        self.datapath_entry.pack(side=LEFT)
        self.datapath_entry.configure(state='disabled')

        self.datapath_browse = Button(self.data_frame, text="Browse...", command=self.train_datapath_dialog)
        self.datapath_browse.pack(side=LEFT)

        self.train_tab.add(self.data_frame)

        # Pre-processing frame
        self.preproc_frame = Labelframe(self.train_tab, text='Pre-processing')
        self.preproc_frame.pack()

        self.sym_intvar = IntVar()
        self.sym_chkbtn = Checkbutton(self.preproc_frame, text="Symmetric modality augmentation",
                                      variable=self.sym_intvar)
        self.sym_chkbtn.grid(row=0, column=0, sticky='w', columnspan=2, pady=self.pad_y)

        self.ss_intvar = IntVar()
        self.ss_chkbtn = Checkbutton(self.preproc_frame, text="Skull stripping (only ISLES18)", variable=self.ss_intvar)
        self.ss_chkbtn.grid(row=1, column=0, columnspan=2, sticky='w', pady=self.pad_y)

        self.train_tab.add(self.preproc_frame)

        # Model name
        self.model_frame = Labelframe(self.train_tab, text='CNN Model')
        self.model_frame.pack()

        self.model_lbl = Label(self.model_frame, text="Name: ")
        self.model_lbl.grid(column=0, row=0, sticky='w')

        self.model_stringvar = StringVar()
        self.model_entry = Entry(self.model_frame, width=30, textvariable=self.model_stringvar,
                                 validate='all', validatecommand=self.validate_model_name)
        self.model_entry.grid(column=1, row=0, sticky='e', columnspan=2, pady=self.pad_y)

        # Pretrained_model
        self.pretrain_lbl = Label(self.model_frame, text="Use pre-trained model: ")
        self.pretrain_lbl.grid(column=0, row=1, columnspan=2, pady=self.pad_y)

        self.pretrain_combo = Combobox(self.model_frame, postcommand=self.validate_update_pretrained_model_list)
        self.pretrain_combo['values'] = ['None', ] + [model_key for model_key in self.models_list]
        self.pretrain_combo.current(0)
        self.pretrain_combo.configure(state='readonly')
        self.pretrain_combo.grid(column=2, row=1, pady=self.pad_y)

        self.train_tab.add(self.model_frame)

        # Errors and start button
        self.button_frame = Frame(self.train_tab)
        self.button_frame.pack()

        self.error_stringvar = StringVar()
        self.error_lbl = Label(self.button_frame, textvariable=self.error_stringvar, foreground="#f44")
        self.error_lbl.pack(side=LEFT)

        self.btn = Button(self.button_frame, text="Start training", command=self.launch_training)
        self.btn.pack(side=RIGHT)

        self.train_tab.add(self.button_frame)

    def validate_selected_test_model(self):
        if self.test_model_stringvar.get() is 'None':
            self.test_errors.update({'model_name': 'none selected'})
        else:
            self.test_errors.pop('model_name', None)
        return True

    def validate_model_name(self):
        if self.model_stringvar.get() in self.models_list:
            self.model_entry.config(foreground="#f44")
            self.train_errors.update({'model_name': 'already in use'})
            self.error_stringvar.set('model_name: "{}" already in use'.format(self.model_stringvar.get()))
        else:
            self.model_entry.config(foreground="#000")
            self.train_errors.pop('model_name', None)
            self.error_stringvar.set('')
        return True

    def train_datapath_dialog(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(title="Select dataset specification file",
                                              filetypes=(("Text File", "*.txt"),))
        self.datapath_entry.configure(state='enabled')
        self.datapath_entry.delete(0, len(self.datapath_entry.get()))
        self.datapath_entry.insert(0, filename)
        self.datapath_entry.configure(state='readonly')
        self.train_errors.pop('data_path', None)

    def test_datapath_dialog(self):
        from tkinter import filedialog
        filename = filedialog.askopenfilename(title="Select dataset specification file",
                                              filetypes=(("Text File", "*.txt"),))
        self.test_datapath_entry.configure(state='enabled')
        self.test_datapath_entry.delete(0, END)
        self.test_datapath_entry.insert(0, filename)
        self.test_datapath_entry.configure(state='readonly')
        self.test_errors.pop('data_path', None)

    def launch_inference(self):
        config_dict = {
            'execution': 'inference',
            'dataset_path': self.test_datapath_entry.get(),
            'skull_stripping': self.test_ss_intvar.get() == 1,
            'pretrained_name': self.test_model_stringvar.get(),
        }  # Symmetric modalities and ct idx order are contained in the model dictionary

        if len(self.test_datapath_entry.get()) <= 0:
            self.test_errors.update({'data_path': 'is empty'})
        else:
            self.test_errors.pop('data_path', None)

        if self.test_model_stringvar.get() == 'None':
            self.test_errors.update({'model_name': 'None selected'})
        else:
            self.test_errors.pop('model_name', None)

        self.validate_selected_test_model()

        if len(self.test_errors) > 0:
            errors = ['{}: {}'.format(k, v) for k, v in self.test_errors.items()]
            self.test_error_stringvar.set(errors[0])
            self.test_error_lbl.config(foreground="#f44")
            return False

        self.test_error_lbl.config(foreground="#000")
        self.test_error_stringvar.set('Running in terminal...')

        self.disable_test_gui()
        self.launch_docker(config_dict)

    def validate_update_pretrained_model_list(self):
        with open(self.storage_path + 'models.txt', 'r') as models_file:
            models_dict = json.loads(models_file.read())  # use `json.dumps` to do the reverse
            self.models_list = [k for k in models_dict]

        self.pretrain_combo.configure(state='normal')
        self.pretrain_combo['values'] = ['None', ] + [model_key for model_key in self.models_list]
        self.pretrain_combo.configure(state='readonly')

        self.test_pretrain_combo.configure(state='normal')
        self.test_pretrain_combo['values'] = ['None', ] + [model_key for model_key in self.models_list]
        self.test_pretrain_combo.configure(state='readonly')
        return True

    def launch_training(self):
        # -------------------------
        # Get config and check ERRORS
        # -------------------------
        config_dict = {
            'execution': 'training',
            'dataset_path': self.datapath_entry.get(),
            'symmetric_modalities': self.sym_intvar.get() == 1,
            'skull_stripping': self.ss_intvar.get() == 1,
            'model_name': self.model_entry.get(),
            'pretrained_name': self.pretrain_combo.get(),
        }

        if len(self.datapath_entry.get()) <= 0:
            self.train_errors.update({'data_path': 'is empty'})
        else:
            self.train_errors.pop('data_path', None)

        if len(self.model_stringvar.get()) <= 0:
            self.train_errors.update({'model_name': 'is empty'})
        else:
            self.train_errors.pop('model_name', None)

        if len(self.train_errors) > 0:
            errors = ['{}: {}'.format(k, v) for k, v in self.train_errors.items()]
            self.error_lbl.config(foreground="#f44")
            self.error_stringvar.set(errors[0])
            return False

        self.error_lbl.config(foreground="#000")
        self.error_stringvar.set('Running in terminal...')  # NO ERRORS!

        self.disable_train_gui()
        self.launch_docker(config_dict)

    def launch_docker(self, config_dict):

        dataset_pathfile = config_dict['dataset_path']
        dataset_path, dataset_filename = os.path.split(dataset_pathfile)

        # Write to config FILE as json
        config_dict['dataset_path'] = '/storage/dataset/{}'.format(dataset_filename)
        with open(self.storage_path + 'config.txt', 'w') as config_file:
            config_file.write(json.dumps(config_dict))  # use `json.loads` to do the reverse
        print(config_dict)

        with open('storage/make.txt', 'r') as config_file:
            make_dict = json.loads(config_file.read())

        if not make_dict['initialized']:
            docker_build_command = "docker build --tag=strokect ."
            self.proc = subprocess.Popen(docker_build_command, universal_newlines=True, shell=True)
            self.wait_for_process()
            self.proc.wait()
            make_dict.update({'initialized': True})

        with open('storage/make.txt', 'w') as config_file:
            config_file.write(json.dumps(make_dict))

        cwd = os.getcwd()
        docker_command = "docker run -it --ipc=host --runtime=nvidia --volume={0}/results/:/results " \
                         "--volume={0}/storage/:/storage --volume={1}:/storage/dataset/ strokect".format(cwd, dataset_path)

        self.proc = subprocess.Popen(docker_command, universal_newlines=True, shell=True)
        self.wait_for_process()

    def disable_test_gui(self):
        self.notebook.tab(0, state="disabled")
        self.test_btn.config(state='disabled')

    def disable_train_gui(self):
        self.notebook.tab(1, state="disabled")
        self.btn.config(state='disabled')

    def enable_gui(self):
        self.notebook.tab(1, state="normal")
        self.btn.config(state='normal')
        self.notebook.tab(0, state="normal")
        self.test_btn.config(state='normal')

        self.error_stringvar.set(' ')  # NO ERRORS!
        self.test_error_stringvar.set(' ')  # NO ERRORS!

    def wait_for_process(self):
        if self.proc.poll() is None:
            self.window.after(66, self.wait_for_process)
        else:
            self.enable_gui()


OtherGUI()

I LEARNED ABOUT THE SPEECH TO SPEECH
REFERENCE - "Textless Speech-to-Speech Translation on Real Data"
IT requires a lot of research 
i tried to install the libraries but it didnt work correctly it is giving the same error



"  Getting requirements to build editable ... error
  error: subprocess-exited-with-error

  × Getting requirements to build editable did not run successfully.
  │ exit code: 1
  ╰─> [17 lines of output]
      Traceback (most recent call last):
        File "C:\Users\princ\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 353, in <module>
          main()
        File "C:\Users\princ\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 335, in main
          json_out['return_val'] = hook(**hook_input['kwargs'])
        File "C:\Users\princ\AppData\Local\Programs\Python\Python310\lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 132, in get_requires_for_build_editable
          return hook(config_settings)
        File "C:\Users\princ\AppData\Local\Temp\pip-build-env-_q7it9ro\overlay\Lib\site-packages\setuptools\build_meta.py", line 448, in get_requires_for_build_editable
          return self.get_requires_for_build_wheel(config_settings)
        File "C:\Users\princ\AppData\Local\Temp\pip-build-env-_q7it9ro\overlay\Lib\site-packages\setuptools\build_meta.py", line 325, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=['wheel'])
        File "C:\Users\princ\AppData\Local\Temp\pip-build-env-_q7it9ro\overlay\Lib\site-packages\setuptools\build_meta.py", line 295, in _get_build_requires
          self.run_setup()
        File "C:\Users\princ\AppData\Local\Temp\pip-build-env-_q7it9ro\overlay\Lib\site-packages\setuptools\build_meta.py", line 311, in run_setup
          exec(code, locals())
        File "<string>", line 246, in <module>
      OSError: [WinError 1314] A required privilege is not held by the client: '..\\examples' -> 'fairseq\\examples'
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error



which becomes a road block for i tried to fetch data of parallel english and hindi and it was successfully. link of the data set is "https://huggingface.co/datasets/google/fleurs/tree/main/data"


further i tried to learn encoder and decoder which is remmonded by gemini

i tried to find the models for speech and i found one "hubert" 

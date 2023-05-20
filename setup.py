setup(
    name='mytSNE',
    version=1.00,
    description='CSE185 tSNE Project',
    author='Siddharth Gaywala',
    author_email='sgaywala@ucsd.edu',
    #packages=find_packages(),
    entry_points={
        "console_scripts": [
            "mytSNE=mytSNE.mytSNE:main"
        ]
    }
    
    
)
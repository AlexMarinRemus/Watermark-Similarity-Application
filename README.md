# CS for the Humanities - Watermarks

## Table of contents

- [Background](#background)
  - [Brief Overview of the System](#brief-overview-of-the-system)
  - [Constraints on the System](#constraints-on-the-system)
- [Running the System](#running-the-system)
  - [Requirements](#requirements)
  - [Running the Command Line System](#running-the-command-line-system)
  - [Running the Graphical User Interface](#running-the-graphical-user-interface)
- [Building the Database](#building-the-database)
  - [Building the Database Automatically](#building-the-database-automatically)
  - [Building the Database Manually](#building-the-database-manually)
- [Dataset Labeling and Splitting](#dataset-labeling-and-splitting)
  - [Labeling the Dataset](#labeling-the-dataset)
  - [Splitting the Dataset](#splitting-the-dataset)
- [System Evaluation](#system-evaluation)
  - [Evaluation Details File](#evaluation-details-file)
- [Testing](#testing)
- [GitLab Setup](#gitlab-setup)
  - [GitLab Wiki](#gitlab-wiki)
  - [Issues](#issues)
  - [Merge Requests](#merge-requests)
  - [GitLab LFS](#gitlab-lfs)
- [Explanation of File Structure](#explanation-of-file-structure)
- [References](#references)


## Background

Watermarks are pictures with symbols, letters, initials, and/or drawings that can identify the producers of the paper [1]. They can be seen by shining light through the paper from a certain angle. In our dataset they are catalogued as scans of the watermarked paper, which we call untraced watermarks. The other possibility is scans of tracings of watermarks.

Researchers often come across old documents that have a certain watermark. These watermarks can be useful because they can provide information about the origin of that document [1] [2]. But in order to understand where or when the watermark may come from, the researchers must contact specialists who then manually go through an archive of watermarks. This can be a tedious process. Our project seeks to solve this problem by producing a system that will allow researchers to find similar watermarks programmatically, which will help them get more information on the watermark they input.

The dataset being used comes courtesy of German Museum of Books and Writing, which has the most extensive collection of watermarks in Europe. The museum has made many of their watermarks available through [the Bernstein project](https://www.memoryofpaper.eu/BernsteinPortal/appl_start.disp). The results of this software project will be a prototype for a tool that can be used by the museum to aid with analysis of their watermarks.

### Brief Overview of the System

There are two main parts to this project: the watermark pipeline which analyzes watermarks and calculates their similarity, and the interface which is used to interact with this system.

For the pipeline there are three main steps. In the harmonization step, the input image is processed such that the watermark is isolated as best as possible. The feature extraction steps involve representing this isolated watermark as a vector. Finally, similarity matching is comparing this vector representation of the watermark to a database of other watermarks and ranking which are the most similar.

There are two main interfaces for this project, one is through the command line and the other is through a graphical user interface (GUI). How these are run can be found in ['Running the System'](#running-the-system).

### Constraints on the System

It is important to note that this in order for our system to work properly, we make several assumptions about the input that the system recieves. First, and most importantly, it is assumed that the watermark image that is input contains a watermark, and that the image is cropped around the watermark. It is also expected that the user provides input regarding whether or not the watermark is traced or not. For further information on how these inputs work, refer to the ['Running the System'](#running-the-system) section.

## Running the System

There are two ways of running the system, through the command line and through a graphical user interface (GUI). Running the system in the command line is faster, but incorporates less user interaction which therefore leads to less flexibility in the harmonization. This can lead to less accuracy. On the other hand, the GUI incorporates user interaction which allows for more accurately harmonized watermarks.

### Requirements

To run the system, [Python 3.10](https://www.python.org/downloads/) or above should be installed. After that, all the necessary libraries should be installed for this project to work. These libraries are included in the requirements.txt file. To install them run:

```console
pip install -r requirements.txt
```

### Running the Command Line System

The similarity matching algorithm can be executed through the command line by running `main.py`. There are several arguments that can be provided: the path to the input watermark image, the path to the database `.pkl` file, the number of similar images to output, and a flag indicating whether the watermark is traced or not. It is also possible to flag debug mode. Run the system through the terminal with:

```shell
python main.py --input_path <path/to/image.jpg> --db_path <path/to/database.pkl> --number_of_output <integer> --is_traced --debug
```

If the paths are not provided as arguments, the system runs with a default input path of `dataset_images/Training/1_1.jpg` and a default database path of `database/db.pkl`. Additionally, if the number of output images is not specified it defaults to 20. In order for the system to run, it is also necessary to have all required libraries installed, as described in the [Requirements](#requirements) section of this README.

Once the command line prompt has been entered correctly, the system will take a little time to process the input and will then output a ranking of similar watermarks. The number of watermarks that are output corresponds to the number specified by the user. Each line in the output ranking will be formatted as follows:

```
<rank number>, <path/to/similar_watermark>, <similarity measure>
```

### Running the Graphical User Interface

In order to run the graphical user interface, `app.py` must be run, as follows:

```shell
python app.py
```

This will start up the GUI locally on port 5000. In order to view the system, open a Chrome browser and go to the URL `http://localhost:5000/` or `http://localhost:5000/input`. Both of these paths will lead to the starting page of the GUI. From there, the user can input the watermark they want to analyze and go through each step in the GUI to get the best output possible. To shut down localhost, go to the terminal running it and press `Ctrl+C` to quit.

The GUI incorporates far more user interaction to ensure that the analysis of the watermark is ideal. Therefore the GUI has several pages that involve customizing thresholding and denoising values, as well as being able to draw on the image to add/erase lines. To make these steps as clear as possible, there is a short description of the purposeon the left side of each page. In the output page of the GUI, the user can see their input image, the results of the harmonization pipeline, and all of the similar images that the system finds.

Note that currently this system only runs on a development server, since deployment was not part of the expectations for this project.

## Building the Database

Although any database can be provided to the system, it is expected that the database is formatted in a certain way. For this reason, it can be very useful to build the database through the methods provided by this system. There are two ways of doing this, one is automatic and the other is manual. As can be expected the automatic method is far faster, but is also less accurate when it comes to harmonizing watermarks. On the other hand, the manual system makes use of user input to make the harmonization process as accurate as possible, but thereby also takes much longer.

It is important to note that when building a database the data is appended to the database with the specified name if it already exists. However, if the file with the specified name does not exist, it creates a new file.

### Building the Database Automatically

To build the database automatically, run the `build_database.py` file. This takes three arguments, the input path to the directory of images to build the database from, the name of the database to create/append to, and the --is_traced flag which states if the images should be processed as traced images or not. The command will therefore look as follows:

```shell
python build_database.py --input_path <path/to/images> --db_name <database_name> --is_traced
```

Note that for the db_name should exclude the `.pkl` at the end, and should also exclude the `database/` path at the start (since it will always be generated in the database folder). If the `--input_path` and `--db_name` arguments are not specified they will default to the input path being `dataset_images/Training` and a database name of `db`. Once the command is run, it will take a little while for all of the images to be processed, and once complete will result in a `.pkl` file with the desired name being generated in the `database` folder.

### Building the Database Manually

There is also an option to build the database manually, through the GUI. This is beneficial because it allows the user to specify for each image if it is traced, and also allows the user to customize the harmonization of each image that is being generated. This process allows for the database to be built with more accurately harmonized images. To do this, `app.py` needs to be run as follows:

```shell
python app.py
```

Then the user must navigate to `http://localhost:5000/build_database`. This will lead to the start screen of building the database. Here the user can specify the path for the directory of images that are to be processed. The directory must be within the working directory, otherwise it cannot be accessed. It is also assumed the name of the directory is unique and does not appear elsewhere in the working directory. The GUI will then go through each image in the directory and allow the user to customize the harmonization process, after which the resulting features are saved. Once this process is complete for all images in the directory, all of the features are saved into the database file, which will be generated or appended to.

Note that the features are not saved to the database until all images have been processed. If this building process is interrupted before all of the images from the path are processed, then the database will not be generated.

## Dataset Labeling and Splitting

To implement and evaluate the recognition system, we have extracted and manually annotated a subset of 500 images from the database provided by the German Museum of Books and Writing. Within these images there are 151 distinct watermarks, each with 2 to 5 different examples of the watermark. After labeling, the data was shuffled and divided into two parts, each having a different purpose in the development of the tool.

### Labeling the Dataset

Each image in the dataset has been assigned a distinctive tag, following the format `X_Y.jpg/.png`. In this tag structure, the `X` represents the index of the watermark in the entire dataset, while the `Y` denotes the index associated with a specific variation of the watermark.

### Splitting the Dataset

The shuffled dataset of labeled images was divided into two sets through an 85-15 split, with 85% of the images allocated for training and 15% for evaluation. The training set was used in implementing and determining parameters for the harmonization, feature extraction and similarity matching steps. It was also used to build the database. The evaluation set was employed to assess the performance of the system in identifying similar watermarks to a given image.

## System Evaluation

To evaluate the watermark recognition system, execute the `evaluation.py` file. Two arguments are required for this: the path to the evaluation set and the path to the `.pkl` file representing the database used for evaluation. The structure of the command will be as follows:

```
python evaluation.py --file_path <path/to/evaluation/set> --db_path <path/to/database.pkl>
```

If the `--file_path` and `--db_path` arguments are not specified, their default values are set to `dataset_images/Evaluation/` and `database/auto_db.pkl`, respectively. The `Evaluation` directory consists of two subfolders, `evaluation_traced` and `evaluation_untraced`, containing the traced and untraced images. This distribution allows the system to apply the appropriate harmonization operations for each type of watermarks during the evaluation process. The `auto_db.pkl` file is the database obtained by running the `build_database.py` file on the training set.

The purpose of the evaluation is to assess how well the system performs in processing an unknown watermark image and retrieving similar watermarks from a trained dataset. This is achieved by iterating through the evaluation set, applying the watermark processing pipeline to each image, and then ranking the images in the database based on their similarity scores with the processed result. Different pictures of the same watermark are called matches. The system determines the ranks of all matches for each image being evaluated and counts the number of all images that have at least one match within the top ranks. A match is considered top-ranked if its position is among the first 10 for traced watermarks, and among the first 15 for untraced watermarks.

Three accuracy scores are returned, representing the overall accuracy, the accuracy for traced watermarks, and the accuracy for untraced watermarks. Each score represents the percentage of images from the evaluation set with at least one top-ranked match that was found in the database.

Because the 500 images of watermarks initally selected for training and evaluation contained unclear images that were difficult to distinguish even for the human viewer, `auto_db_clear.pkl` was created by selecting 200 other pictures from the database provided by the German Museum of Books and Writing. They were split into `Training` and `Evaluation`, with four images corresponding to each of the 50 watermarks in this dataset. The division follows a 75-25 split, with 75% of the images being used to build the database automatically, and 25% of the pictures being used for evaluation. Within the evaluation set, each picture is associated with two images of the same type (either traced or untraced), as well as one image of the opposite type, all of them being variations of the same watermark. This database was created in order to assess how impactful is the choice of a dataset for the overall performance of the system.

### Evaluation Details File

Running the evaluation will also generate a `.txt` file named `evaluation/evaluation_ranking_details_{database_path}.txt`. It contains an overview of all images including their name, type and number of top-ranked matches. Additionally, for each match from the database, it provides details regarding the rank number, name and type. The `.txt` file is formatted as follows:

```
img-name=<X_Y.jpg/.png>, top-ranked-match-count=<number/of/top/ranked/matches>
 - rank=<position_in_ranking>, name=<X_Y.jpg/.png>, type=<traced/untraced>
 ...
 - rank=<position_in_ranking>, name=<X_Y.jpg/.png>, type=<traced/untraced>
```

The `image-name` corresponds to the annotated name of the input image from the evaluation set and `top-ranked-match-count` corresponds to the total number of top-ranked matches for the input image. Then, for each match in the database, `rank` represents the position it occupies within the ranked list, `name` represents its annotated name, and `type` indicates whether it is a traced or an untraced watermark. This file provides an understanding of how the system performs when matching watermarks of the same type as opposed to matching watermarks of different types, also offering insights into the general distribution of rank positions.

## Testing

In order to run tests, the following command can be run:

```shell
python -m pytest
```

This will automatically go through all of the files in the working directory, and run the test files. To generate a coverage report run the following:

```shell
python -m pytest --cov --cov-report=term --cov-report=xml:coverage.xml --cov-report=html
```

Note that the front end tests (and only the front end tests) are flaky (although they do pass a significant majority of the time). This is primarily because they are dependent on the timing of loading. Loading can change depending on the hardware of the computer. Sometimes, running the front end tests the first time will lead to some failures, and then running the second time will work (we assume this is because of caching). For this reason it may be desirable to exclude front end tests from the tests, to do this run the following:

```shell
python -m pytest --ignore=testing/front_end_testing
```

## GitLab Setup

Below the GitLab setup for this project is described, including the GitLab wiki, the issues, and GitLab LFS.

### GitLab Wiki

Much of the documentation of this project is in the GitLab wiki, including the code of conduct, the research documents, the sprint retrospectives, agendas, and meeting notes.

- The code of conduct outlines expectations for how we are to work as a team, this is primarily a reference for the teammembers themselves.
- The research documents include notes on several research papers for different sections of the pipeline. These have been included for personal reference, but also in case any future developers may want to reference techniques that have been used, or that could be used. These research papers can be relatively extensive.
- The sprint retrospectives break down the work that each member has done in each sprint of this project. They also debrief the sprint, and discuss the goal of the sprint, strong points of the sprint, problems that have been run into, and future adjustments.
- The agendas and meeting notes are fairly straightforward, and include relevant notes taken during various meetings throughout the course of this project.

### Issues

GitLab issues break down the various parts of this project into manageable tasks, that we associate Merge Requests to. Each issue has a description of the task that is to be complete in that issue. If applicable, there is also the user story associated with the issue. There is a To Do list, which describes tasks that need to be completed. Finally there is the done criteria which describes how we know the specific task is done.

Issues also have time estimates, time spent, weights, milestones, and assignees.

- Time estimates and time spent metrics are used to determine how much effort might need to go into a task, and how much time has been spent on a task.
- Weights are assigned based on their importance to the project on a scale of 1 to 10, 10 being most important.
- Tasks that have a weight of 10 must be completed, and tasks with a weight of 1 have very low priority.
- Milestones track which sprints a task has been assigned to. Since several milestones cannot be put to a single task, the most recent one appears on the issue.
- Assignees are those members that have worked on an issue.

Issues also have associated labels that are used to designate the various categories a task may belong to. For example, User Interface, Testing, Database, etc. There are also "TODO" and "Doing" labels that allow us to see which issues are currently being worked on and which we plan on working on soon.

### Merge Requests

Merge requests contain the code that is worked on. They usually relate to a specific issue. Comments on the merge requests are made by those that have not worked on the code in that merge request. There is also a milestone associated to merge requests. Merge requests also sometimes have a Checklist, that ensures that several necessary steps, such as ensuring there are no style errors, are addressed. If relevant there is also more information on details relevant to a merge request. However, the majority of documentation pertaining to a merge request will be found in its corresponding issue.

### GitLab LFS

LFS is used by us to store the large files of this project. Specifically, this includes the database `.pkl` file, as well as the dataset of images that we have been using. These are included in the GitLab because they are important to the running of the system, but are too large to store normally.

## Explanation of File Structure

This project has several directories, here we will provide a brief explanation on the main parts of the project directory. There are also a number of configuration files for things such as testing, coverage and git LFS. The names of these start with a ".".

- The `database` directory stores the `.pkl` files, and any other database code that was used that is not useful for the user to call directly.
- The `dataset_images` directory stores the dataset of watermark images that were used during this project. They are split into two subdirectories, `clear_dataset`, which contains only images that are clear, and `original_dataset`, which contains a mix of clear and unclear images. Both of these have two subdirectories, `Training` and `Evaluation`, that are used to test the system.
- The `evaluation` directory stores the output `.txt` files that result from running the evaluation. There are also some files that were manually created that show other evaluation results and a subdirectory for the GUI evaluation recordings.
- The `feature_extraction` directory stores all files related to feature extraction.
- The `harmonization` directory stores all files related to harmonization.
- The `similarity_comparison` directory stores all files related to comparing similarity between watermarks.
- The `static` directory stores files that are used by for the GUI that are not `html` files. These files includes `js` files, `css` files, and any relevant images.
- The `templates` directory stores all the html files used for the GUI.
- The `testing` directory stores all tests for the files within the directory. The subdirectory `front_end_testing` stores specifically front end tests.
- The `app.py` file is the file used to run the GUI, as described in ['Running the Graphical User Interface'](#running-the-graphical-user-interface) and ['Building the Database Manually'](#building-the-database-manually)
- The `build_database.py` file is run to automatically build the database, as described in ['Building the Database Automatically'](#building-the-database-automatically).
- The `evaluation.py` file is run to evaluate the system, as described in [System Evaluation](#system-evaluation)
- The `main.py` file is run to automatically run the system in the command line, as described in ['Running the Command Line System'](#running-the-command-line-system)

## References

[1] L. Müller, “Understanding Paper: Structures, Watermarks, and a Conservator’s Passion,” Harvard Art Museums, May 07, 2021. Accessed: Apr. 29, 2023. Available [https://harvardartmuseums.org/article/understanding-paper-structures-watermarks-and-a-conservator-s-passion#:~:text=Sometimes%20the%20pattern%20is%20more](https://harvardartmuseums.org/article/understanding-paper-structures-watermarks-and-a-conservator-s-passion#:~:text=Sometimes%20the%20pattern%20is%20more)

[2] “Watermarks \& the History of Paper,” Research Group on Manuscript Evidence, Jul. 05, 2020. Available [https://manuscriptevidence.org/wpme/watermarks-and-the-history-of-paper/](https://manuscriptevidence.org/wpme/watermarks-and-the-history-of-paper/) (accessed May 01, 2023).

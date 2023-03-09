# from subprocess import Popen, PIPE
import subprocess


def delete_qjobs(job_min: int, job_max: int):
    """
    Function deletes all jobs with an id between job_min and
    job_max

    args:
        - job_min - id of smallest job which should be stopped
        - job_max - id of biggest job which should be stopped
    """
    jobs = []
    for idx in range(job_max - job_min):
        jobs.append(job_min + idx)

    for job in jobs:
        r = subprocess.run(
            # ['qdel %s ' % job],
            ['scancel %s ' % job],
            shell=True
        )


if __name__ == '__main__':
    # specify job id on cluster
    job_min = 918056
    job_max = 918115

    delete_qjobs(job_min, job_max)

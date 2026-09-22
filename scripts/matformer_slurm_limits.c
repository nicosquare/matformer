/* Read controller limits when sacctmgr or a per-user QoS usage row is absent.
 * Compile against the installed Slurm headers/library; no scheduler mutations.
 */
#include <stdio.h>
#include <string.h>
#include <slurm/slurm.h>
#include <slurm/slurmdb.h>

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    slurm_init(NULL);
    assoc_mgr_info_request_msg_t request = {0};
    request.flags = ASSOC_MGR_INFO_FLAG_QOS | ASSOC_MGR_INFO_FLAG_ASSOC;
    assoc_mgr_info_msg_t *info = NULL;
    if (slurm_load_assoc_mgr_info(&request, &info)) return 1;
    list_itr_t *it = slurm_list_iterator_create(info->qos_list);
    slurmdb_qos_rec_t *q;
    while ((q = slurm_list_next(it))) {
        if (!strcmp(q->name, argv[1]))
            printf("QOS|%u|%u|%u|%u|%u|%u\n", q->max_jobs_pu,
                   q->max_submit_jobs_pu, q->max_jobs_pa,
                   q->max_submit_jobs_pa, q->grp_jobs, q->grp_submit_jobs);
    }
    slurm_list_iterator_destroy(it);
    it = slurm_list_iterator_create(info->assoc_list);
    slurmdb_assoc_rec_t *a;
    while ((a = slurm_list_next(it))) {
        if (!a->user || !*a->user || !strcmp(a->user, argv[2]))
            printf("ASSOC|%u|%u|%s|%u|%u|%u|%u\n", a->id,
                   a->parent_id, a->user ? a->user : "", a->max_jobs,
                   a->max_submit_jobs, a->grp_jobs, a->grp_submit_jobs);
    }
    slurm_list_iterator_destroy(it);
    slurm_free_assoc_mgr_info_msg(info);
    return 0;
}

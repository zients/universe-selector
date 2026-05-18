alter table report_artifacts rename to report_artifacts_old;

create table report_artifacts (
    run_id varchar not null references run_log(run_id),
    format varchar not null check (format in ('markdown', 'json')),
    content varchar not null,
    primary key (run_id, format)
);

insert into report_artifacts(run_id, format, content)
select run_id, format, content from report_artifacts_old;

drop table report_artifacts_old;

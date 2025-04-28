FROM public.ecr.aws/docker/library/ruby:3.2.2

RUN apt-get update && apt-get install -y build-essential ruby-dev

RUN gem install bundler -v 2.6.5

WORKDIR /opt/rb-ckmeans
COPY . .
RUN bundle install -j 12

ENTRYPOINT ["bundle", "exec"]

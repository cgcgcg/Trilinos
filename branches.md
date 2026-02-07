```mermaid
gitGraph
   branch master
   commit
   branch develop
   checkout develop
   commit
   commit
   checkout master
   merge develop id: "master merge 1"
   checkout develop
   commit
   checkout master
   merge develop id: "master merge 2"
```
